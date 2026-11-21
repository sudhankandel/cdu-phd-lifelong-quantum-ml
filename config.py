
import os
import random

import numpy as np
import torch

# ==========================================
# 1. REPRODUCIBILITY
# ==========================================
PRIMARY_SEED = 4826
SEED = int(os.environ.get("LL_SEED", PRIMARY_SEED))


def set_seed(seed=SEED):
    if seed is None:
        raise ValueError(
            "Set SEED using the last four digits "
            "of your reference number."
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(
        True, warn_only=True
    )
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

# Fraction of training pool used for validation
VALIDATION_RATIO = 0.20

NUM_CLASSES = 6
CLASSES_PER_TASK = 2
NUM_TASKS = 3


# ==========================================
# 3. TRAINING
# ==========================================

# Provisional defaults
# Update after validation-based tuning
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

# Hyperparameter search (validation only)
TUNING_LEARNING_RATES = [0.0001, 0.0005, 0.001]
TUNING_EPOCHS = [10, 20]
TUNING_BATCH_SIZES = [8]

BACKBONE = "resnet18"
PRETRAINED = True


# ==========================================
# 4. QUANTUM
# ==========================================
NUM_QUBITS = 6
QUANTUM_LAYERS = 2
SHOTS = None
FINITE_SHOTS = 1024

LOGIT_SCALE = 5.0


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

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


if __name__ == "__main__":
    set_seed()

    print("Seed:", SEED)
    print("Device:", get_device())
