import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

sys.path.append("/content/drive/MyDrive/task4")

from models.resnet_cifar import CIFARResNet18
from data.cifar100_unknowns import CIFAR100UnknownDataset


# ============================================================
# Configuration
# ============================================================

SEED = 6304

DATA_ROOT = "/content/drive/MyDrive/task4/data"

PROSER_CHECKPOINT = (
    "/content/drive/MyDrive/task4/cache/proser_best.pt"
)

BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_DUMMY_CLASSES = 5
NUM_WORKERS = 2

NEAR_CLASSES = [
    "bus",
    "pickup_truck",
    "motorcycle",
    "tractor",
    "wolf",
    "fox",
    "leopard",
    "camel"
]

FAR_CLASSES = [
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
# Reproducibility
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


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


# ============================================================
# Evaluation transform
# ============================================================

eval_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    )
])


# ============================================================
# Load PROSER checkpoint
# ============================================================

checkpoint = torch.load(
    PROSER_CHECKPOINT,
    map_location=device
)

print(
    "Loaded PROSER checkpoint:"
)

print(
    "Checkpoint epoch:",
    checkpoint.get("epoch", "unknown")
)

print(
    "Checkpoint validation accuracy:",
    checkpoint.get("val_accuracy", "unknown")
)


# ============================================================
# Create 15-class PROSER model
#
# 10 known classes
# + 5 dummy classifiers
# ============================================================

model = CIFARResNet18(
    num_classes=NUM_CLASSES + NUM_DUMMY_CLASSES
).to(device)


# ============================================================
# Load checkpoint
# ============================================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# Load CIFAR-10
# ============================================================

base_dataset = datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    download=True,
    transform=None
)

targets = np.array(
    base_dataset.targets
)

indices = np.arange(
    len(base_dataset)
)


# ============================================================
# Recreate official 90/10 stratified split
# ============================================================

train_indices, val_indices = train_test_split(

    indices,

    test_size=0.10,

    random_state=SEED,

    stratify=targets
)


# ============================================================
# CIFAR-10 validation dataset
# ============================================================

cifar10_train_eval = datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    download=False,
    transform=eval_transform
)

val_dataset = Subset(
    cifar10_train_eval,
    val_indices
)


# ============================================================
# CIFAR-10 complete test set
# ============================================================

cifar10_test = datasets.CIFAR10(
    root=DATA_ROOT,
    train=False,
    download=True,
    transform=eval_transform
)


# ============================================================
# CIFAR-100 unknown datasets
# ============================================================

near_dataset = CIFAR100UnknownDataset(
    root=DATA_ROOT,
    class_names=NEAR_CLASSES,
    transform=eval_transform
)

far_dataset = CIFAR100UnknownDataset(
    root=DATA_ROOT,
    class_names=FAR_CLASSES,
    transform=eval_transform
)


# ============================================================
# Verify unknown set sizes
# ============================================================

print()
print("Unknown set sizes:")
print("Near:", len(near_dataset))
print("Far :", len(far_dataset))

assert len(near_dataset) == 800, (
    f"Expected 800 near unknowns, "
    f"got {len(near_dataset)}"
)

assert len(far_dataset) == 800, (
    f"Expected 800 far unknowns, "
    f"got {len(far_dataset)}"
)


# ============================================================
# DataLoaders
# ============================================================

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    cifar10_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

near_loader = DataLoader(
    near_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

far_loader = DataLoader(
    far_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# Extract logits
# ============================================================

def extract_logits(model, loader):

    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            logits = model(images)

            all_logits.append(
                logits.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

    return (
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0)
    )


# ============================================================
# PROSER MLS
#
# Directly comparable with Vanilla/GCSC:
#
# u_MLS(x) = - max_k z_k
#
# where k is restricted to the 10 known classes.
#
# Larger score = more novel.
# ============================================================

def mls_score(logits):

    known_logits = logits[
        :, :NUM_CLASSES
    ]

    return -known_logits.max(
        dim=1
    ).values


# ============================================================
# PROSER placeholder response
#
# Strongest dummy classifier response.
# ============================================================

def dummy_score(logits):

    dummy_logits = logits[
        :, NUM_CLASSES:
    ]

    return dummy_logits.max(
        dim=1
    ).values


# ============================================================
# CIFAR-10 validation
# ============================================================

print()
print("Evaluating CIFAR-10 validation set...")

val_logits, val_labels = extract_logits(
    model,
    val_loader
)

val_known_logits = val_logits[
    :, :NUM_CLASSES
]

val_predictions = val_known_logits.argmax(
    dim=1
)

val_accuracy = (
    val_predictions == val_labels
).float().mean().item()


print(
    f"PROSER validation accuracy: "
    f"{val_accuracy:.4f}"
)


# ============================================================
# CIFAR-10 test
# ============================================================

print()
print("Evaluating CIFAR-10 test set...")

test_logits, test_labels = extract_logits(
    model,
    test_loader
)

test_known_logits = test_logits[
    :, :NUM_CLASSES
]

test_predictions = test_known_logits.argmax(
    dim=1
)

test_accuracy = (
    test_predictions == test_labels
).float().mean().item()


print(
    f"PROSER CIFAR-10 test accuracy: "
    f"{test_accuracy:.4f}"
)


# ============================================================
# Unknown evaluation
# ============================================================

print()
print("Evaluating near unknowns...")

near_logits, _ = extract_logits(
    model,
    near_loader
)

print(
    "Evaluating far unknowns..."
)

far_logits, _ = extract_logits(
    model,
    far_loader
)


# ============================================================
# MLS scores
# ============================================================

val_mls = mls_score(
    val_logits
)

test_mls = mls_score(
    test_logits
)

near_mls = mls_score(
    near_logits
)

far_mls = mls_score(
    far_logits
)


# ============================================================
# Placeholder scores
# ============================================================

val_dummy = dummy_score(
    val_logits
)

test_dummy = dummy_score(
    test_logits
)

near_dummy = dummy_score(
    near_logits
)

far_dummy = dummy_score(
    far_logits
)


# ============================================================
# Print score statistics
# ============================================================

def print_statistics(name, scores):

    print(
        f"{name}: "
        f"mean={scores.mean().item():.4f}, "
        f"std={scores.std().item():.4f}, "
        f"min={scores.min().item():.4f}, "
        f"max={scores.max().item():.4f}"
    )


print()
print("============================================================")
print("MLS SCORE STATISTICS")
print("============================================================")

print_statistics(
    "CIFAR-10 validation",
    val_mls
)

print_statistics(
    "CIFAR-10 test",
    test_mls
)

print_statistics(
    "CIFAR-100 near",
    near_mls
)

print_statistics(
    "CIFAR-100 far",
    far_mls
)


print()
print("============================================================")
print("DUMMY / PLACEHOLDER SCORE STATISTICS")
print("============================================================")

print_statistics(
    "CIFAR-10 validation",
    val_dummy
)

print_statistics(
    "CIFAR-10 test",
    test_dummy
)

print_statistics(
    "CIFAR-100 near",
    near_dummy
)

print_statistics(
    "CIFAR-100 far",
    far_dummy
)


# ============================================================
# Save extracted outputs
# ============================================================

OUTPUT_DIR = (
    "/content/drive/MyDrive/task4/results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


torch.save(

    {
        "val_logits": val_logits,
        "val_labels": val_labels,

        "test_logits": test_logits,
        "test_labels": test_labels,

        "near_logits": near_logits,

        "far_logits": far_logits,

        "val_mls": val_mls,
        "test_mls": test_mls,

        "near_mls": near_mls,
        "far_mls": far_mls,

        "val_dummy": val_dummy,
        "test_dummy": test_dummy,

        "near_dummy": near_dummy,
        "far_dummy": far_dummy,

        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,

        "near_classes": NEAR_CLASSES,
        "far_classes": FAR_CLASSES

    },

    os.path.join(
        OUTPUT_DIR,
        "proser_outputs.pt"
    )
)


print()
print(
    "Saved PROSER evaluation outputs to:"
)

print(
    os.path.join(
        OUTPUT_DIR,
        "proser_outputs.pt"
    )
)