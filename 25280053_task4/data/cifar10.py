# ============================================================
# task4/data/cifar10.py
# ============================================================

import os
import numpy as np
import torch

from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from .make_splits import make_stratified_split


SEED = 6304


# ------------------------------------------------------------
# CIFAR-10 normalization
# ------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ------------------------------------------------------------
# Transforms
# ------------------------------------------------------------

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


# ------------------------------------------------------------
# CIFAR-100 fixed unknown groups
# ------------------------------------------------------------

NEAR_UNKNOWN_CLASSES = [
    "bus",
    "pickup_truck",
    "motorcycle",
    "tractor",
    "wolf",
    "fox",
    "leopard",
    "camel",
]

FAR_UNKNOWN_CLASSES = [
    "bottle",
    "bowl",
    "chair",
    "clock",
    "keyboard",
    "mushroom",
    "sunflower",
    "wardrobe",
]


class CIFAR100UnknownDataset(Dataset):
    """
    Fixed CIFAR-100 test examples used only for final OSR evaluation.

    CIFAR-100 training data is NEVER used.
    """

    def __init__(self, root, class_names, transform=None):
        self.transform = transform

        dataset = datasets.CIFAR100(
            root=root,
            train=False,
            download=True,
            transform=None
        )

        self.dataset = dataset

        class_to_idx = {
            name: idx
            for idx, name in enumerate(dataset.classes)
        }

        selected_indices = []

        for class_name in class_names:
            if class_name not in class_to_idx:
                raise ValueError(
                    f"{class_name} not found in CIFAR-100 classes."
                )

            class_idx = class_to_idx[class_name]

            indices = [
                i for i, label in enumerate(dataset.targets)
                if label == class_idx
            ]

            # CIFAR-100 test set contains 100 examples/class
            selected_indices.extend(indices)

        self.indices = selected_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):

        original_idx = self.indices[index]

        image, label = self.dataset[original_idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_datasets(root="./data"):
    """
    Returns:

        train_dataset
        val_dataset
        test_dataset
        near_unknown_dataset
        far_unknown_dataset
    """

    # --------------------------------------------------------
    # Official CIFAR-10 training set
    # --------------------------------------------------------

    base_train = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=None
    )

    # --------------------------------------------------------
    # Create stratified 90/10 split
    # --------------------------------------------------------

    train_indices, val_indices = make_stratified_split(
        base_train.targets,
        seed=SEED
    )

    # --------------------------------------------------------
    # Separate dataset objects so that only training receives
    # augmentation.
    # --------------------------------------------------------

    train_full = datasets.CIFAR10(
        root=root,
        train=True,
        download=False,
        transform=train_transform
    )

    val_full = datasets.CIFAR10(
        root=root,
        train=True,
        download=False,
        transform=eval_transform
    )

    train_dataset = Subset(
        train_full,
        train_indices
    )

    val_dataset = Subset(
        val_full,
        val_indices
    )

    # --------------------------------------------------------
    # Complete official CIFAR-10 test set
    # --------------------------------------------------------

    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=eval_transform
    )

    # --------------------------------------------------------
    # Fixed CIFAR-100 TEST unknowns
    # --------------------------------------------------------

    near_unknown_dataset = CIFAR100UnknownDataset(
        root=root,
        class_names=NEAR_UNKNOWN_CLASSES,
        transform=eval_transform
    )

    far_unknown_dataset = CIFAR100UnknownDataset(
        root=root,
        class_names=FAR_UNKNOWN_CLASSES,
        transform=eval_transform
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        near_unknown_dataset,
        far_unknown_dataset,
    )