import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torchvision.datasets import STL10

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# import values from config file
from configs.config import (
    SEED,
    VAL_RATIO,
    TEST_SUBSET_SIZE,
    RESULTS_DIR,
)

# reproducibility of the same random number each time 
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Load STL10 dataset 
def load_stl10(root=None):
    if root is None:
        root = PROJECT_ROOT / "data" / "stl10"

    train_dataset = STL10(
        root=str(root),
        split="train",
        download=True
    )

    test_dataset = STL10(
        root=str(root),
        split="test",
        download=True
    )

    return train_dataset, test_dataset

# split the data equally between 80/20
def create_train_val_split(dataset):

    indices = np.arange(len(dataset))
    labels = np.asarray(dataset.labels)

    train_indices, val_indices = train_test_split(
        indices,
        test_size=VAL_RATIO,
        random_state=SEED,
        shuffle=True,
        stratify=labels
    )

    return train_indices, val_indices


# Class-balanced test subset
def create_balanced_test_subset(dataset):

    labels = np.asarray(dataset.labels)

    rng = np.random.default_rng(SEED)

    classes = np.unique(labels)

    # Desired number per class 500/10 = 50
    base_count = TEST_SUBSET_SIZE // len(classes)

    selected_indices = []
    records = []

    for class_id in classes:

        class_indices = np.where(labels == class_id)[0]

        rng.shuffle(class_indices)

        num_to_select = min(
            base_count,
            len(class_indices)
        )

        chosen = class_indices[:num_to_select]

        selected_indices.extend(chosen.tolist())

        for idx in chosen:
            records.append({
                "test_index": int(idx),
                "class": int(class_id),
            })

    # If there are still fewer than 500 images fill from remaining test images.
    if len(selected_indices) < TEST_SUBSET_SIZE:

        remaining = np.setdiff1d(
            np.arange(len(dataset)),
            np.array(selected_indices)
        )

        rng.shuffle(remaining)

        needed = TEST_SUBSET_SIZE - len(selected_indices)

        extra = remaining[:needed]

        for idx in extra:
            selected_indices.append(int(idx))

            records.append({
                "test_index": int(idx),
                "class": int(labels[idx]),
            })

    selected_indices = np.array(selected_indices)

    rng.shuffle(selected_indices)

    df = pd.DataFrame(records)

    # Reorder dataframe according to selected order
    order = {
        idx: position
        for position, idx in enumerate(selected_indices)
    }

    df["order"] = df["test_index"].map(order)
    df = df.sort_values("order").drop(columns=["order"])

    return selected_indices, df



# Main
def main():

    set_seed(SEED)

    train_dataset, test_dataset = load_stl10()

    print("STL-10 training images:", len(train_dataset))
    print("STL-10 test images:", len(test_dataset))

    # Train / validation split
    train_indices, val_indices = create_train_val_split(train_dataset)

    print()
    print("Training samples:", len(train_indices))
    print("Validation samples:", len(val_indices))

    # Verify stratification
    train_labels = np.asarray(train_dataset.labels)[train_indices]
    val_labels = np.asarray(train_dataset.labels)[val_indices]

    print("\nTraining class counts:")
    print(np.bincount(train_labels))

    print("\nValidation class counts:")
    print(np.bincount(val_labels))

    # Save split indices
    np.save(
        RESULTS_DIR / "train_indices.npy",
        train_indices)

    np.save(
        RESULTS_DIR / "val_indices.npy",
        val_indices)


    # Test subset
    test_indices, subset_df = create_balanced_test_subset(
        test_dataset)

    print()
    print("Selected test images:", len(test_indices))

    print("\nTest subset class counts:")
    print(subset_df["class"].value_counts().sort_index())

    # Save identifiers
    subset_df.to_csv(
        RESULTS_DIR / "subset_ids.csv",
        index=False)

    np.save(
        RESULTS_DIR / "test_indices.npy",
        test_indices)

    print(
        f"\nSaved test identifiers to: "
        f"{RESULTS_DIR / 'subset_ids.csv'}")


if __name__ == "__main__":
    main()