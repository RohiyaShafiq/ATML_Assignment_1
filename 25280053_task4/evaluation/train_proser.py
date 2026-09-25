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
from methods.manifold_mixup import manifold_mixup
from methods.proser import PROSERLoss


# ============================================================
# Configuration
# ============================================================

SEED = 6304

DATA_ROOT = "/content/drive/MyDrive/task4/data"

VANILLA_CHECKPOINT = (
    "/content/drive/MyDrive/task4/results/vanilla_best.pt"
)

PROSER_CHECKPOINT = (
    "/content/drive/MyDrive/task4/cache/proser_best.pt"
)

BATCH_SIZE = 128
NUM_EPOCHS = 50

LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

NUM_CLASSES = 10
NUM_DUMMY_CLASSES = 5

BETA = 1.0
GAMMA = 0.1

MIXUP_ALPHA = 2.0

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
# Training transform
# ============================================================

train_transform = transforms.Compose([

    transforms.RandomCrop(
        32,
        padding=4
    ),

    transforms.RandomHorizontalFlip(),

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
# Load CIFAR-10 training partition
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
# 90/10 stratified split
# ============================================================

train_indices, val_indices = train_test_split(

    indices,

    test_size=0.10,

    random_state=SEED,

    stratify=targets
)


# ============================================================
# Training dataset
# ============================================================

train_dataset_full = datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    download=False,
    transform=train_transform
)

train_dataset = Subset(
    train_dataset_full,
    train_indices
)


# ============================================================
# Validation dataset
# ============================================================

val_dataset_full = datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    download=False,
    transform=eval_transform
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


print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(val_dataset)
)


# ============================================================
# Load Vanilla model
# ============================================================

vanilla_model = CIFARResNet18(
    num_classes=NUM_CLASSES
).to(device)


# ============================================================
# Load selected Vanilla checkpoint
# ============================================================

checkpoint = torch.load(
    VANILLA_CHECKPOINT,
    map_location=device
)


vanilla_model.load_state_dict(
    checkpoint["model_state_dict"]
)


vanilla_model.eval()


print(
    "Loaded Vanilla checkpoint:"
)

print(
    "Vanilla validation accuracy:",
    checkpoint["val_accuracy"]
)


# ============================================================
# Create PROSER model
#
# 10 known classifiers
# + 5 dummy classifiers
# = 15 outputs
# ============================================================

model = CIFARResNet18(
    num_classes=NUM_CLASSES + NUM_DUMMY_CLASSES
).to(device)


# ============================================================
# Initialize PROSER known classifiers from Vanilla
#
# First 10 classifier rows are copied from Vanilla.
# Last 5 rows remain randomly initialized.
# ============================================================

with torch.no_grad():

    model.backbone.fc.weight[:NUM_CLASSES].copy_(
        vanilla_model.backbone.fc.weight
    )

    model.backbone.fc.bias[:NUM_CLASSES].copy_(
        vanilla_model.backbone.fc.bias
    )


print(
    "Initialized PROSER with Vanilla checkpoint."
)

print(
    "Known classifiers:",
    NUM_CLASSES
)

print(
    "Dummy classifiers:",
    NUM_DUMMY_CLASSES
)


# ============================================================
# Loss
# ============================================================

proser_loss = PROSERLoss(

    num_known_classes=NUM_CLASSES,

    num_dummy_classes=NUM_DUMMY_CLASSES,

    beta=BETA,

    gamma=GAMMA
)


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
# Cosine schedule
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=NUM_EPOCHS
)


# ============================================================
# Forward to representation after layer2
# ============================================================

def forward_to_layer2(model, x):

    b = model.backbone

    x = b.conv1(x)

    x = b.bn1(x)

    x = b.relu(x)

    x = b.maxpool(x)

    x = b.layer1(x)

    x = b.layer2(x)

    return x


# ============================================================
# Forward from representation after layer2
# through layer3, layer4 and classifier
# ============================================================

def forward_from_layer2(model, x):

    b = model.backbone

    x = b.layer3(x)

    x = b.layer4(x)

    x = b.avgpool(x)

    x = torch.flatten(
        x,
        1
    )

    logits = b.fc(x)

    return logits


# ============================================================
# Validation
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

            # Only known-class logits are used
            # for CIFAR-10 validation accuracy.
            known_logits = logits[:, :NUM_CLASSES]

            predictions = known_logits.argmax(
                dim=1
            )

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
    os.path.dirname(PROSER_CHECKPOINT),
    exist_ok=True
)


for epoch in range(NUM_EPOCHS):

    model.train()

    total_loss = 0.0
    total_cls = 0.0
    total_placeholder = 0.0
    total_data_placeholder = 0.0

    total_samples = 0


    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()


        # ====================================================
        # Split mini-batch into two equal halves
        # ====================================================

        half = images.size(0) // 2

        images_classifier = images[:half]
        labels_classifier = labels[:half]

        images_mixup = images[half:]
        labels_mixup = labels[half:]


        # ====================================================
        # Classifier-placeholder half
        # ====================================================

        logits_classifier = model(
            images_classifier
        )

        known_logits_classifier = (
            logits_classifier[:, :NUM_CLASSES]
        )

        dummy_logits_classifier = (
            logits_classifier[:, NUM_CLASSES:]
        )


        # ====================================================
        # Classifier-placeholder loss
        # ====================================================

        (
            loss_main,
            loss_cls,
            loss_cp,
            _
        ) = proser_loss(

            known_logits_classifier,

            dummy_logits_classifier,

            labels_classifier
        )


        # ====================================================
        # Manifold Mixup half
        #
        # Representation:
        # after layer2, before layer3
        # ====================================================

        h_mix = forward_to_layer2(
            model,
            images_mixup
        )


        (
            mixed_h,
            labels_a,
            labels_b,
            lam
        ) = manifold_mixup(

            h_mix,

            labels_mixup,

            alpha=MIXUP_ALPHA
        )


        # ====================================================
        # Continue forward pass from layer3
        # ====================================================

        mixed_logits = forward_from_layer2(
            model,
            mixed_h
        )


        mixed_known_logits = (
            mixed_logits[:, :NUM_CLASSES]
        )

        mixed_dummy_logits = (
            mixed_logits[:, NUM_CLASSES:]
        )


        # ====================================================
        # Data-placeholder loss
        # ====================================================

        loss_data_placeholder = (
            proser_loss.data_placeholder_loss(

                mixed_known_logits,

                mixed_dummy_logits
            )
        )


        # ====================================================
        # Total loss
        # ====================================================

        total_batch_loss = (

            loss_main

            + GAMMA * loss_data_placeholder
        )


        # ====================================================
        # Backpropagation
        # ====================================================

        total_batch_loss.backward()

        optimizer.step()


        # ====================================================
        # Statistics
        # ====================================================

        batch_size = images.size(0)

        total_loss += (
            total_batch_loss.item()
            * batch_size
        )

        total_cls += (
            loss_cls.item()
            * batch_size
        )

        total_placeholder += (
            loss_cp.item()
            * batch_size
        )

        total_data_placeholder += (
            loss_data_placeholder.item()
            * batch_size
        )

        total_samples += batch_size


    # ========================================================
    # Scheduler
    # ========================================================

    scheduler.step()


    # ========================================================
    # Average losses
    # ========================================================

    train_loss = (
        total_loss
        / total_samples
    )

    cls_loss_value = (
        total_cls
        / total_samples
    )

    cp_loss_value = (
        total_placeholder
        / total_samples
    )

    dp_loss_value = (
        total_data_placeholder
        / total_samples
    )


    # ========================================================
    # Validation
    # ========================================================

    val_accuracy = evaluate(
        model,
        val_loader
    )


    # ========================================================
    # Logging
    # ========================================================

    print(
        f"Epoch [{epoch + 1:03d}/{NUM_EPOCHS}] "
        f"Loss: {train_loss:.4f} "
        f"CE: {cls_loss_value:.4f} "
        f"CP: {cp_loss_value:.4f} "
        f"DP: {dp_loss_value:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )


    # ========================================================
    # Save best checkpoint
    #
    # Selection uses CIFAR-10 validation accuracy only.
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(

            {
                "model_state_dict":
                    model.state_dict(),

                "val_accuracy":
                    val_accuracy,

                "epoch":
                    epoch + 1,

                "seed":
                    SEED
            },

            PROSER_CHECKPOINT
        )

        print(
            "  Saved best PROSER checkpoint."
        )


# ============================================================
# Finished
# ============================================================

print()

print(
    "PROSER training finished."
)

print(
    "Best validation accuracy:",
    best_val_accuracy
)

print(
    "Checkpoint:",
    PROSER_CHECKPOINT
)