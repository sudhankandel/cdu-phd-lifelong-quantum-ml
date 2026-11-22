from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import SEED, PRIMARY_SEED, set_seed, TRAIN_RATIO


# ------------------------------------------
# Paths
# ------------------------------------------
DATA_DIR = Path("data")

IMAGE_DIR = DATA_DIR / "ISIC2018_Task3_Training_Input"

LABEL_FILE = (
    DATA_DIR / "ISIC2018_Task3_Training_GroundTruth.csv"
)

OUTPUT_DIR = DATA_DIR / "processed" / f"seed{SEED}"


def load_dataset():
    """Load ISIC 2018 Task 3 labels."""

    if not LABEL_FILE.exists():
        raise FileNotFoundError(
            f"Ground-truth file not found: {LABEL_FILE}"
        )

    df = pd.read_csv(LABEL_FILE)

    if "image" not in df.columns:
        raise ValueError("Missing image column")

    class_names = sorted(
        col for col in df.columns if col != "image"
    )

    if len(class_names) != 7:
        raise ValueError(
            f"Expected 7 classes, got {len(class_names)}"
        )

    labels = df[class_names].to_numpy()

    if not np.all(labels.sum(axis=1) == 1):
        raise ValueError(
            "Each image must have exactly one class"
        )

    df["original_class"] = (
        df[class_names].idxmax(axis=1)
    )

    df["image_path"] = df["image"].apply(
        lambda image: str(IMAGE_DIR / f"{image}.jpg")
    )

    missing = [
        path for path in df["image_path"]
        if not Path(path).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images missing. "
            f"Example: {missing[0]}"
        )

    return df, class_names


def create_task_sequence(class_names):
    """Shuffle seven classes and retain six."""

    rng = np.random.RandomState(SEED)

    shuffled = rng.permutation(
        sorted(class_names)
    ).tolist()

    excluded_class = shuffled[-1]
    selected_classes = shuffled[:6]

    tasks = [
        selected_classes[0:2],
        selected_classes[2:4],
        selected_classes[4:6],
    ]

    return tasks, excluded_class


def prepare_dataset():
    """Create reproducible lifelong-learning splits."""

    if SEED is None:
        raise ValueError("Set SEED in config.py")

    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, class_names = load_dataset()

    tasks, excluded_class = create_task_sequence(
        class_names
    )

    selected_classes = [
        class_name
        for task in tasks
        for class_name in task
    ]

    df = df[
        df["original_class"].isin(selected_classes)
    ].copy()

    # NOTE: HAM10000 has several images per lesion and the ISIC CSV has no
    # lesion id, so an image-level split may place near-duplicates in both
    # train and test. Acknowledge this limitation in the report.

    # Fixed global labels for all three tasks
    class_to_id = {
        name: index
        for index, name in enumerate(selected_classes)
    }

    df["label"] = df["original_class"].map(
        class_to_id
    )

    df["task"] = df["label"] // 2 + 1

    # Stratified split across the six selected classes
    train_df, test_df = train_test_split(
        df,
        train_size=TRAIN_RATIO,
        random_state=SEED,
        stratify=df["label"],
        shuffle=True,
    )

    train_df = train_df.sort_values(
        "image"
    ).reset_index(drop=True)

    test_df = test_df.sort_values(
        "image"
    ).reset_index(drop=True)

    # Save splits
    columns = [
        "image",
        "image_path",
        "original_class",
        "label",
        "task",
    ]

    train_df[columns].to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    test_df[columns].to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
    )

    # Save reproducibility metadata
    metadata = {
        "seed": SEED,
        "is_primary_seed": SEED == PRIMARY_SEED,
        "train_ratio": TRAIN_RATIO,
        "excluded_class": excluded_class,
        "tasks": tasks,
        "class_to_id": class_to_id,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nDataset preparation complete!")
    print(f"FIRST LINE OF REPORT -> Seed: {SEED}; task sequence: {tasks}; excluded: {excluded_class}")
    print("Seed:", SEED)
    print("Excluded class:", excluded_class)

    for index, task in enumerate(tasks, start=1):
        print(f"Task {index}: {task}")

    print("\nTraining samples:", len(train_df))
    print("Testing samples:", len(test_df))

    print("\nTraining distribution:")
    print(
        train_df.groupby(
            ["task", "original_class"]
        ).size()
    )

    print("\nTesting distribution:")
    print(
        test_df.groupby(
            ["task", "original_class"]
        ).size()
    )

    print("\nFiles saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    prepare_dataset()
