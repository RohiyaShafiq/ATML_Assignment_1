import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.cifar10 import get_datasets
from models.resnet_cifar import CIFARResNet18


SEED = 6304

BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

DATA_ROOT = "/content/drive/MyDrive/task4/data"
CHECKPOINT_DIR = "/content/drive/MyDrive/task4/results"

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "vanilla_best.pt"
)


def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# Validation accuracy
@torch.no_grad()
def evaluate_accuracy(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    for images, targets in loader:

        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == targets
        ).sum().item()

        total += targets.size(0)

    return correct / total


# Main training
def main():

    set_seed()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    (
        train_dataset,
        val_dataset,
        _,
        _,
        _
    ) = get_datasets(DATA_ROOT)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )


    model = CIFARResNet18(
        num_classes=10
    ).to(device)

    # Random initialization is already provided by
    # resnet18(weights=None).

    criterion = nn.CrossEntropyLoss()

    # SGD
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Cosine decay
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    best_val_accuracy = -1.0

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        total = 0

        for images, targets in train_loader:

            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(
                logits,
                targets
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * targets.size(0)
            )

            total += targets.size(0)

        scheduler.step()

        train_loss = running_loss / total

        val_accuracy = evaluate_accuracy(
            model,
            val_loader,
            device
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
            f"Loss: {train_loss:.4f} "
            f"Val Acc: {val_accuracy:.4f} "
            f"LR: {current_lr:.6f}"
        )

        # ----------------------------------------------------
        # Save highest validation accuracy checkpoint
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": best_val_accuracy,
                    "seed": SEED,
                },
                CHECKPOINT_PATH
            )

            print(
                f"  Saved checkpoint "
                f"(Val Acc = {best_val_accuracy:.4f})"
            )

    print()
    print("Training complete.")
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )
    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )


if __name__ == "__main__":
    main()