
"""
Hybrid quantum-classical model.

    Image -> frozen ResNet-18 -> 512 features
          -> Linear(512, 6) -> pi*tanh -> RY angle encoding (6 qubits)
          -> StronglyEntanglingLayers -> <Z_i> on each qubit
          -> x LOGIT_SCALE -> optional Linear(6,6) readout -> 6 class logits

HOW SIX CLASSES ARE READ OUT FROM <= 8 QUBITS
  * Qubits      : 6 (limit is 8).
  * Encoding    : qubit i receives RY(angle_i), angle_i = pi*tanh(w_i . f + b_i),
                  so angles are bounded in [-pi, pi]. One learned feature per qubit.
  * Ansatz      : StronglyEntanglingLayers (2 layers x 6 qubits x 3 rotations =
                  36 trainable weights). Entangling CNOT rings let information
                  from every encoded feature reach every qubit's measurement.
  * Measurement : Pauli-Z expectation <Z_i> on every qubit, each in [-1, 1].
  * Class map   : qubit i  <->  global class id i (one dedicated qubit per class),
                  score_i = LOGIT_SCALE * <Z_i>. These go to softmax cross-entropy.
                  Limitation: scores are bounded, hence the fixed LOGIT_SCALE.
  * Shots       : shots=None -> exact expectations; shots=1024 -> each <Z_i> is
                  estimated from 1,024 samples (std. error up to ~1/sqrt(1024)=0.031),
                  trained with parameter-shift gradients.

Trainable parameters of the head: 512*6+6 (projection) + 36 (circuit) = 3,114.
"""

import torch
import torch.nn as nn
import pennylane as qml

from config import (SEED, NUM_CLASSES, NUM_QUBITS, QUANTUM_LAYERS,
                    LOGIT_SCALE, set_seed)
from classical_model import ClassicalResNet18


class QuantumClassifier(nn.Module):
    def __init__(self, n_qubits=NUM_QUBITS, n_layers=QUANTUM_LAYERS, shots=None):
        super().__init__()
        if n_qubits != 6 or NUM_CLASSES != 6:
            raise ValueError("Six qubits are used for six class outputs.")
        if n_qubits > 8:
            raise ValueError("At most 8 qubits are allowed.")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots

        # Seeded device so finite-shot sampling is reproducible.
        self.qdevice = qml.device("default.qubit", wires=n_qubits, seed=SEED)
        diff_method = "backprop" if shots is None else "parameter-shift"

        def circuit(inputs, weights):
            # inputs: [batch, n_qubits] (parameter broadcasting) or [n_qubits]
            for wire in range(n_qubits):
                qml.RY(inputs[..., wire], wires=wire)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        qnode = qml.QNode(circuit, self.qdevice, interface="torch",
                          diff_method=diff_method)
        if shots is not None:
            # Newer PennyLane: shots set on the QNode; older: on the device.
            if hasattr(qml, "set_shots"):
                qnode = qml.set_shots(qnode, shots=shots)
            else:
                self.qdevice = qml.device("default.qubit", wires=n_qubits,
                                          shots=shots, seed=SEED)
                qnode = qml.QNode(circuit, self.qdevice, interface="torch",
                                  diff_method=diff_method)
        self.circuit = qnode

        self.weights = nn.Parameter(
            0.01 * torch.randn((n_layers, n_qubits, 3), dtype=torch.float64))

    def forward(self, angles):
        """angles: [batch, 6] -> logits [batch, 6] (whole batch in one call)."""
        angles = angles.to(torch.float64)
        if self.shots is None:
            # Exact mode: one broadcast circuit call for the whole batch.
            out = self.circuit(angles, self.weights)
            logits = torch.stack(list(out), dim=-1)      # [batch, 6]
        else:
            # Finite-shot mode: PennyLane's parameter-shift rule does not
            # support gradients through broadcasted tapes (PennyLane issue
            # #4462), so execute one sample at a time (mathematically identical).
            rows = [torch.stack(list(self.circuit(a, self.weights)))
                    for a in angles]
            logits = torch.stack(rows)                    # [batch, 6]
        return LOGIT_SCALE * logits


class HybridQuantumResNet18(nn.Module):
    def __init__(self, pretrained=True, n_qubits=NUM_QUBITS,
                 n_layers=QUANTUM_LAYERS, shots=None, readout=False):
        super().__init__()
        classical = ClassicalResNet18(NUM_CLASSES, pretrained)
        self.backbone = classical.backbone.to("cpu")
        self.projection = nn.Linear(512, n_qubits).to(dtype=torch.float64)
        self.quantum_head = QuantumClassifier(n_qubits, n_layers, shots)
        self.readout = nn.Linear(n_qubits, NUM_CLASSES, dtype=torch.float64) if readout else None

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def extract_features(self, images):
        with torch.no_grad():
            return self.backbone(images.to("cpu"))

    def forward(self, images):
        features = self.extract_features(images).to(torch.float64)
        angles = torch.pi * torch.tanh(self.projection(features))
        scores = self.quantum_head(angles)
        return self.readout(scores) if self.readout is not None else scores


def create_quantum_model(pretrained=True, shots=None, readout=False):
    set_seed(SEED)
    return HybridQuantumResNet18(pretrained, NUM_QUBITS, QUANTUM_LAYERS, shots, readout)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    """Checks: broadcasting == per-sample loop, gradients flow, param count."""
    set_seed(SEED)
    head = QuantumClassifier(shots=None)
    proj = nn.Linear(512, 6).double()
    x = torch.randn(4, 512, dtype=torch.float64)
    angles = torch.pi * torch.tanh(proj(x))

    batched = head(angles)
    looped = torch.stack([head(a.unsqueeze(0))[0] for a in angles])
    assert batched.shape == (4, 6)
    assert torch.allclose(batched, looped, atol=1e-8), "broadcast mismatch"

    loss = nn.CrossEntropyLoss()(batched, torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert torch.isfinite(head.weights.grad).all()
    assert torch.isfinite(proj.weight.grad).all()

    n = sum(p.numel() for p in proj.parameters()) + head.weights.numel()
    print("Head trainable parameters:", n)
    assert n == 3114

    shot_head = QuantumClassifier(shots=1024)
    out = shot_head(angles.detach())
    print("Finite-shot logits (should be noisy):\n", out)
    print("Exact logits:\n", head(angles.detach()))
    # Verify the optional 42-parameter readout and its gradients.
    readout = nn.Linear(6, 6, dtype=torch.float64)
    fresh_angles = torch.pi * torch.tanh(proj(x))
    readout_loss = nn.CrossEntropyLoss()(readout(head(fresh_angles)), torch.tensor([0, 1, 2, 3]))
    readout_loss.backward()
    assert readout.weight.grad is not None and torch.isfinite(readout.weight.grad).all()
    assert proj.weight.grad is not None and torch.isfinite(proj.weight.grad).all()
    assert n + sum(p.numel() for p in readout.parameters()) == 3156
    print("Readout-enabled trainable parameters: 3156")
    # Verify the finite-shot backward path on an unbroadcasted sample.
    shot_proj = nn.Linear(512, 6).double()
    shot_angles = torch.pi * torch.tanh(shot_proj(x[:1]))
    shot_loss = nn.CrossEntropyLoss()(shot_head(shot_angles), torch.tensor([0]))
    shot_loss.backward()
    assert shot_head.weights.grad is not None and torch.isfinite(shot_head.weights.grad).all()
    assert shot_proj.weight.grad is not None and torch.isfinite(shot_proj.weight.grad).all()
    print("Finite-shot backward verification successful!")
    print("Quantum model verification successful!")


if __name__ == "__main__":
    main()
