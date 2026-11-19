
"""
Architecture:
    Image -> Frozen ResNet-18 -> 512 features
          -> Linear(512, 6) -> tanh -> angle encoding
          -> 6-qubit VQC -> 6 class logits

The VQC uses:
    Encoding: RY angle encoding
    Ansatz: StronglyEntanglingLayers
    Measurement: Pauli-Z expectation values
    Readout: Six expectation values = six class scores

This is a single-head classifier for all three tasks.
"""

import torch
import torch.nn as nn
import pennylane as qml

from config import (
    SEED,
    NUM_CLASSES,
    NUM_QUBITS,
    QUANTUM_LAYERS,
    set_seed,
)

from classical_model import ClassicalResNet18


class QuantumClassifier(nn.Module):
    """
    Differentiable variational quantum classification head.

    Each of six qubits produces one expectation value,
    which is interpreted as a class logit.
    """

    def __init__(
        self,
        n_qubits=NUM_QUBITS,
        n_layers=QUANTUM_LAYERS,
        shots=None,
    ):
        super().__init__()

        if n_qubits != 6 or NUM_CLASSES != 6:
            raise ValueError(
                "This implementation uses six qubits "
                "for six classification outputs."
            )

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots

        # Exact simulation for normal training.
        # Finite-shot mode can use parameter-shift
        # differentiation in later experiments.
        diff_method = (
            "best" if shots is None
            else "parameter-shift"
        )

        self.device = qml.device(
            "default.qubit",
            wires=n_qubits,
            shots=shots,
        )

        @qml.qnode(
            self.device,
            interface="torch",
            diff_method=diff_method,
        )
        def circuit(inputs, weights):

            # ENCODING:
            # Each projected CNN feature is encoded
            # as an RY rotation on one qubit.
            for wire in range(n_qubits):
                qml.RY(inputs[wire], wires=wire)

            # ANSATZ:
            # Trainable rotations and entangling gates.
            qml.StronglyEntanglingLayers(
                weights,
                wires=range(n_qubits),
            )

            # MEASUREMENT AND CLASS MAPPING:
            # Six Pauli-Z expectation values produce
            # six scores in [-1, 1].
            # Score i corresponds to global class ID i.
            # CrossEntropyLoss accepts these raw logits.
            return [
                qml.expval(qml.PauliZ(wire))
                for wire in range(n_qubits)
            ]

        self.circuit = circuit

        weight_shape = (
            n_layers,
            n_qubits,
            3,
        )

        self.weights = nn.Parameter(
            0.01 * torch.randn(
                weight_shape,
                dtype=torch.float64,
            )
        )

    def forward(self, angles):
        """
        Args:
            angles: [batch_size, 6] CPU tensor

        Returns:
            Quantum logits: [batch_size, 6]

        Per-sample execution is used for compatibility
        and transparent differentiation.
        """

        outputs = []

        for sample in angles:
            measurements = self.circuit(
                sample.to(torch.float64),
                self.weights,
            )

            logits = torch.stack(measurements)
            outputs.append(logits)

        return torch.stack(outputs)


class HybridQuantumResNet18(nn.Module):
    """
    Frozen ResNet-18 + projection + quantum classifier.

    The ResNet backbone is shared architecturally
    with the classical baseline.
    """

    def __init__(
        self,
        pretrained=True,
        n_qubits=NUM_QUBITS,
        n_layers=QUANTUM_LAYERS,
        shots=None,
    ):
        super().__init__()

        classical = ClassicalResNet18(
            num_classes=NUM_CLASSES,
            pretrained=pretrained,
        )

        self.backbone = classical.backbone

        # Keep the frozen backbone on CPU for
        # straightforward PennyLane compatibility.
        self.backbone.to("cpu")

        # 512 CNN features -> 6 quantum inputs
        self.projection = nn.Linear(
            512,
            n_qubits,
        ).to(dtype=torch.float64)

        self.quantum_head = QuantumClassifier(
            n_qubits=n_qubits,
            n_layers=n_layers,
            shots=shots,
        )

    def train(self, mode=True):
        super().train(mode)

        # Preserve frozen BatchNorm statistics.
        self.backbone.eval()

        return self

    def extract_features(self, images):
        """
        Extract frozen ResNet features on CPU.
        """

        with torch.no_grad():
            features = self.backbone(
                images.to("cpu")
            )

        return features

    def forward(self, images):
        """
        Returns six class logits for each image.
        """

        features = self.extract_features(images)

        features = features.to(torch.float64)

        # Bound rotation angles to [-pi, pi].
        angles = torch.pi * torch.tanh(
            self.projection(features)
        )

        logits = self.quantum_head(angles)

        return logits


def create_quantum_model(
    pretrained=True,
    shots=None,
):
    """Create a reproducibly initialised hybrid model."""

    set_seed(SEED)

    model = HybridQuantumResNet18(
        pretrained=pretrained,
        n_qubits=NUM_QUBITS,
        n_layers=QUANTUM_LAYERS,
        shots=shots,
    )

    return model


def count_parameters(model):
    """Count total and trainable parameters."""

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


def main():

    print("=" * 50)
    print(" Hybrid Quantum-Classical Model")
    print("=" * 50)

    print("Seed:", SEED)
    print("Quantum device: default.qubit (CPU)")
    print("Qubits:", NUM_QUBITS)
    print("Quantum layers:", QUANTUM_LAYERS)
    print("Shots: None (exact simulation)")

    model = create_quantum_model(
        pretrained=True,
        shots=None,
    )

    total, trainable = count_parameters(model)

    print("\nParameters:")
    print(f"Total: {total:,}")
    print(f"Trainable: {trainable:,}")
    print(f"Frozen: {total - trainable:,}")

    # Small batch for initial verification
    images = torch.randn(
        2, 3, 224, 224
    )

    model.eval()

    with torch.no_grad():
        logits = model(images)

    print("\nOutput shape:", logits.shape)
    print("Output dtype:", logits.dtype)
    print("Output scores:\n", logits)

    assert logits.shape == (2, NUM_CLASSES)
    assert torch.isfinite(logits).all()

    # Verify that gradients reach both
    # the projection and quantum parameters.
    model.train()

    logits = model(images)

    targets = torch.tensor([0, 1])

    loss = nn.CrossEntropyLoss()(
        logits,
        targets,
    )

    loss.backward()

    assert model.projection.weight.grad is not None
    assert model.quantum_head.weights.grad is not None

    assert torch.isfinite(
        model.projection.weight.grad
    ).all()

    assert torch.isfinite(
        model.quantum_head.weights.grad
    ).all()

    print("\nLoss:", loss.item())
    print(
        "Projection gradient norm:",
        model.projection.weight.grad.norm().item()
    )
    print(
        "Quantum gradient norm:",
        model.quantum_head.weights.grad.norm().item()
    )

    print("\nQuantum model verification successful!")


if __name__ == "__main__":
    main()
