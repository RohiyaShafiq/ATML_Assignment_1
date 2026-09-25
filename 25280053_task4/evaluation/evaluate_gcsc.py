import os
import sys
import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append("/content/drive/MyDrive/task4")

from models.resnet_cifar import CIFARResNet18
from data.cifar100_unknowns import CIFAR100UnknownDataset


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = "/content/drive/MyDrive/task4/data"

CHECKPOINT_PATH = (
    "/content/drive/MyDrive/task4/cache/gcsc_best.pt"
)

OUTPUT_DIR = (
    "/content/drive/MyDrive/task4/cache/gcsc"
)

BATCH_SIZE = 128

NUM_CLASSES = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CIFAR-10 normalization
# ============================================================

CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616
)


eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    )
])


# ============================================================
# Unknown classes
# ============================================================

NEAR_UNKNOWN_CLASSES = [
    "bus",
    "pickup_truck",
    "motorcycle",
    "tractor",
    "wolf",
    "fox",
    "leopard",
    "camel"
]

FAR_UNKNOWN_CLASSES = [
    "bottle",
    "bowl",
    "chair",
    "clock",
    "keyboard",
    "mushroom",
    "sunflower",
    "wardrobe"
]


# ============================================================
# Load model
# ============================================================

model = CIFARResNet18(
    num_classes=NUM_CLASSES
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Loaded GCSC checkpoint from epoch:",
    checkpoint["epoch"]
)

print(
    "Validation accuracy:",
    checkpoint["val_accuracy"]
)


# ============================================================
# CIFAR-10 test set
# ============================================================

test_dataset = datasets.CIFAR10(
    root=DATA_ROOT,
    train=False,
    download=True,
    transform=eval_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Unknown datasets
# ============================================================

near_dataset = CIFAR100UnknownDataset(
    DATA_ROOT,
    NEAR_UNKNOWN_CLASSES,
    transform=eval_transform
)

far_dataset = CIFAR100UnknownDataset(
    DATA_ROOT,
    FAR_UNKNOWN_CLASSES,
    transform=eval_transform
)


near_loader = DataLoader(
    near_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

far_loader = DataLoader(
    far_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MLS score
#
# u_MLS = -max_k z_k
#
# Larger value = more novel
# ============================================================

def mls_score(logits):

    return -logits.max(
        dim=1
    ).values


# ============================================================
# Extract logits
# ============================================================

def extract_logits(loader):

    outputs = []

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(DEVICE)

            logits = model(images)

            outputs.append(
                logits.cpu()
            )

    return torch.cat(outputs)


# ============================================================
# Extract
# ============================================================

test_logits = extract_logits(
    test_loader
)

near_logits = extract_logits(
    near_loader
)

far_logits = extract_logits(
    far_loader
)


# ============================================================
# Accuracy
# ============================================================

correct = (
    test_logits.argmax(dim=1)
    == torch.tensor(test_dataset.targets)
).sum().item()

test_accuracy = (
    correct / len(test_dataset)
)


print()
print(
    f"GCSC CIFAR-10 test accuracy: "
    f"{test_accuracy:.4f}"
)


# ============================================================
# Save logits and MLS scores
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

np.save(
    f"{OUTPUT_DIR}/cifar10_test_logits.npy",
    test_logits.numpy()
)

np.save(
    f"{OUTPUT_DIR}/near_logits.npy",
    near_logits.numpy()
)

np.save(
    f"{OUTPUT_DIR}/far_logits.npy",
    far_logits.numpy()
)

np.save(
    f"{OUTPUT_DIR}/cifar10_test_mls.npy",
    mls_score(test_logits).numpy()
)

np.save(
    f"{OUTPUT_DIR}/near_mls.npy",
    mls_score(near_logits).numpy()
)

np.save(
    f"{OUTPUT_DIR}/far_mls.npy",
    mls_score(far_logits).numpy()
)


print(
    "Saved GCSC outputs to:",
    OUTPUT_DIR
)