import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

import sys
sys.path.append("/content/drive/MyDrive/task4")

from models.resnet_cifar import CIFARResNet18


# ============================================================
# Configuration
# ============================================================

SEED = 6304

DATA_ROOT = "/content/drive/MyDrive/task4/data"
CHECKPOINT_PATH = "/content/drive/MyDrive/task4/cache/gcsc_best.pt"

BATCH_SIZE = 128
NUM_EPOCHS = 100

LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

NUM_CLASSES = 10

NUM_WORKERS = 2


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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
# GCSC TRAINING TRANSFORM
#
# Required:
# Random crop with padding 4
# Random horizontal flip
# RandAugment(num_ops=2, magnitude=9)
# ============================================================

gcsc_train_transform = transforms.Compose([

    transforms.RandomCrop(
        32,
        padding=4
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandAugment(
        num_ops=2,
        magnitude=9
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    )
])


# ============================================================
# Validation transform
# ============================================================

eval_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    )
])


# ============================================================
# Load CIFAR-10
# ============================================================

full_dataset = datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    download=True,
    transform=None
)

targets = np.array(full_dataset.targets)

indices = np.arange(len(full_dataset))


# ============================================================
# Stratified 90/10 split
# ============================================================

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.10,
    random_state=SEED,
    stratify=targets
)


# ============================================================
# Create datasets with correct transforms
# ============================================================

train_dataset_full = datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    download=False,
    transform=gcsc_train_transform
)

val_dataset_full = datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    download=False,
    transform=eval_transform
)


train_dataset = Subset(
    train_dataset_full,
    train_indices
)

val_dataset = Subset(
    val_dataset_full,
    val_indices
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ============================================================
# Model
# ============================================================

model = CIFARResNet18(
    num_classes=NUM_CLASSES
).to(device)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# Cosine learning-rate schedule
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS
)


# ============================================================
# Validation function
# ============================================================

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return correct / total


# ============================================================
# Training
# ============================================================

best_val_accuracy = 0.0


os.makedirs(
    os.path.dirname(CHECKPOINT_PATH),
    exist_ok=True
)


for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0
    total = 0
    correct = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * labels.size(0)
        )

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    scheduler.step()

    train_loss = running_loss / total
    train_accuracy = correct / total

    val_accuracy = evaluate(
        model,
        val_loader
    )

    print(
        f"Epoch [{epoch + 1:03d}/{NUM_EPOCHS}] "
        f"Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )


    # ========================================================
    # Save best checkpoint based ONLY on CIFAR-10 validation
    # accuracy
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_accuracy,
                "epoch": epoch + 1,
                "seed": SEED
            },
            CHECKPOINT_PATH
        )

        print(
            f"  Saved best checkpoint: "
            f"{CHECKPOINT_PATH}"
        )


print()
print("Training finished.")
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)