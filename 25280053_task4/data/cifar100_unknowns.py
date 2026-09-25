import torch
from torch.utils.data import Dataset
from torchvision import datasets


class CIFAR100UnknownDataset(Dataset):

    def __init__(self, root, class_names, transform=None):
        self.transform = transform

        # Load CIFAR-100 test set only
        self.dataset = datasets.CIFAR100(
            root=root,
            train=False,
            download=True
        )

        self.class_names = class_names

        # CIFAR-100 fine-label names
        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(self.dataset.classes)
        }

        # Convert requested class names to CIFAR-100 indices
        self.target_classes = [
            self.class_to_idx[name]
            for name in class_names
        ]

        # Keep only requested classes
        self.indices = [
            i
            for i, label in enumerate(self.dataset.targets)
            if label in self.target_classes
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        image, label = self.dataset[real_idx]

        if self.transform is not None:
            image = self.transform(image)

        # All CIFAR-100 samples in this dataset are unknown
        return image, label