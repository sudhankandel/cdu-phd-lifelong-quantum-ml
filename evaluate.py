"""Balanced accuracy and continual-learning evaluation helpers."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score


def balanced_accuracy(y_true, y_pred) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def average_accuracy(R: np.ndarray, task_index: int) -> float:
    """Mean balanced accuracy across tasks seen through task_index (0-based)."""
    return float(np.mean(R[task_index, :task_index + 1]))


def average_forgetting(R: np.ndarray, task_index: int) -> float:
    """Mean prior peak minus current performance on old tasks (0 at task 1)."""
    if task_index == 0:
        return 0.0
    return float(np.mean([
        np.max(R[:task_index, j]) - R[task_index, j]
        for j in range(task_index)
    ]))
