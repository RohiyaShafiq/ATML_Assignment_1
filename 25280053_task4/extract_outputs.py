# ============================================================
# task4/evaluation/extract_outputs.py
# ============================================================

import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from data.cifar10 import get_datasets
from models.resnet_cifar import CIFARResNet18


# ============================================================
# Paths
# ============================================================

DATA_ROOT = "/content/drive/MyDrive/task4/data"

CHECKPOINT_PATH = (
    "/content/drive/MyDrive/task4/results/"
    "vanilla_best.pt"
)

CACHE_DIR = (
    "/content/drive/MyDrive/task4/cache"
)


BATCH_SIZE = 128


# ============================================================
# Extract features and logits
# ============================================================

@torch.no_grad()
def extract_dataset_outputs(
    model,
    dataset,
    device
):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model.eval()

    all_features = []
    all_logits = []
    all_labels = []

    for images, labels in loader:

        images = images.to(device)

        features, logits = model(
            images,
            return_features=True
        )

        all_features.append(
            features.cpu().numpy()
        )

        all_logits.append(
            logits.cpu().numpy()
        )

        all_labels.append(
            labels.numpy()
        )

    features = np.concatenate(
        all_features,
        axis=0
    )

    logits = np.concatenate(
        all_logits,
        axis=0
    )

    labels = np.concatenate(
        all_labels,
        axis=0
    )

    return features, logits, labels


def main():

    os.makedirs(
        CACHE_DIR,
        exist_ok=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset,
        near_unknown_dataset,
        far_unknown_dataset
    ) = get_datasets(DATA_ROOT)

    # --------------------------------------------------------
    # Load selected checkpoint
    # --------------------------------------------------------

    model = CIFARResNet18(
        num_classes=10
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Loaded checkpoint from epoch:",
        checkpoint["epoch"]
    )

    print(
        "Validation accuracy:",
        checkpoint["val_accuracy"]
    )

    # --------------------------------------------------------
    # Extract outputs
    # --------------------------------------------------------

    datasets_to_extract = {
        "cifar10_train": train_dataset,
        "cifar10_val": val_dataset,
        "cifar10_test": test_dataset,
        "cifar100_near": near_unknown_dataset,
        "cifar100_far": far_unknown_dataset,
    }

    for name, dataset in datasets_to_extract.items():

        print(
            f"\nExtracting {name} "
            f"({len(dataset)} examples)..."
        )

        features, logits, labels = (
            extract_dataset_outputs(
                model,
                dataset,
                device
            )
        )

        np.save(
            os.path.join(
                CACHE_DIR,
                f"{name}_features.npy"
            ),
            features
        )

        np.save(
            os.path.join(
                CACHE_DIR,
                f"{name}_logits.npy"
            ),
            logits
        )

        np.save(
            os.path.join(
                CACHE_DIR,
                f"{name}_labels.npy"
            ),
            labels
        )

        print(
            "Features:",
            features.shape
        )

        print(
            "Logits:",
            logits.shape
        )

    print("\nOutput extraction complete.")


if __name__ == "__main__":
    main()