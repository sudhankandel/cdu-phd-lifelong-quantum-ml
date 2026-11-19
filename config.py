
import os
import random

import numpy as np
import torch

# ==========================================
# 1. REPRODUCIBILITY

SEED = 4826

def set_seed(seed=SEED):
    if seed is None:
        raise ValueError(
            "Set SEED using the last four digits "
            "of your CDU applicant reference number."
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic execution where supported
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# ==========================================
# 2. DATASET
# ==========================================

DATA_DIR = "data"
RESULTS_DIR = "results"
LOG_DIR = "results/logs"

IMAGE_SIZE = 224
TRAIN_RATIO = 0.80
TEST_RATIO = 0.20

NUM_CLASSES = 6
CLASSES_PER_TASK = 2
NUM_TASKS = 3


# ==========================================
# 3. TRAINING CONFIGURATION
# ==========================================

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

BACKBONE = "resnet18"
PRETRAINED = True

# ==========================================
# 4. QUANTUM CONFIGURATION
# ==========================================

NUM_QUBITS = 6
QUANTUM_LAYERS = 2

# None = exact expectation values
SHOTS = None
FINITE_SHOTS = 1024


# ==========================================
# 5. LIFELONG LEARNING
# ==========================================

REHEARSAL_RATIOS = [0.0, 0.05, 0.10, 0.20, 1.0]


# ==========================================
# 6. DEVICE
# ==========================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")

    return torch.device("cpu")


if __name__ == "__main__":
    set_seed()

    print("Seed:", SEED)
    print("Device:", get_device())
    print("Number of tasks:", NUM_TASKS)
    print("Classes per task:", CLASSES_PER_TASK)
    print("Rehearsal ratios:", REHEARSAL_RATIOS)
