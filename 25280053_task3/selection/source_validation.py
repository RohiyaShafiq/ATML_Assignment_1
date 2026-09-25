import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from shared.pacs import PACSDataset, get_transforms

from task3.models.backbone import ResNet18Backbone
from task3.models.classifier_head import ClassifierHead

from task3.methods.erm import ERMMethod
from task3.methods.dan_dg import DANDGMethod
from task3.methods.sam import SAMMethod


# ============================================================
# Constants
# ============================================================

SEED = 6304

SOURCE_DOMAINS = [
    "photo",
    "art_painting",
    "cartoon"
]

NUM_CLASSES = 7

BATCH_SIZE_PER_DOMAIN = 8

EPOCHS = 30
PATIENCE = 5

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Macro-F1
# ============================================================

def macro_f1_from_confusion_matrix(cm):

    f1_scores = []

    for i in range(NUM_CLASSES):

        tp = cm[i, i]

        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        if precision + recall > 0:

            f1 = (
                2 * precision * recall
                / (precision + recall)
            )

        else:

            f1 = 0.0

        f1_scores.append(f1)

    return float(np.mean(f1_scores))


# ============================================================
# Source-domain evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    backbone,
    classifier,
    loader
):

    backbone.eval()
    classifier.eval()

    correct = 0
    total = 0

    cm = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    for images, labels, _ in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        features = backbone(images)

        logits = classifier(features)

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        for true, pred in zip(
            labels.cpu().numpy(),
            predictions.cpu().numpy()
        ):

            cm[true, pred] += 1

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    macro_f1 = macro_f1_from_confusion_matrix(cm)

    return accuracy, macro_f1


# ============================================================
# Domain-balanced source loader
# ============================================================

class CyclingDomainLoader:

    def __init__(self, loaders):

        self.loaders = loaders

        self.num_batches = max(
            len(loader)
            for loader in loaders.values()
        )

    def __iter__(self):

        iterators = {
            domain: iter(loader)
            for domain, loader in self.loaders.items()
        }

        for _ in range(self.num_batches):

            batches = []

            for domain in SOURCE_DOMAINS:

                try:

                    batch = next(
                        iterators[domain]
                    )

                except StopIteration:

                    iterators[domain] = iter(
                        self.loaders[domain]
                    )

                    batch = next(
                        iterators[domain]
                    )

                batches.append(batch)

            images = torch.cat(
                [
                    batch[0]
                    for batch in batches
                ],
                dim=0
            )

            labels = torch.cat(
                [
                    batch[1]
                    for batch in batches
                ],
                dim=0
            )

            domains = []

            for batch in batches:

                domains.extend(batch[2])

            yield images, labels, domains

    def __len__(self):

        return self.num_batches


# ============================================================
# Build source loaders
# ============================================================

def build_source_loaders(split_data):

    train_loaders = {}
    val_loaders = {}

    for domain in SOURCE_DOMAINS:

        train_samples = [
            tuple(sample)
            for sample in
            split_data["sources"][domain]["train"]
        ]

        val_samples = [
            tuple(sample)
            for sample in
            split_data["sources"][domain]["val"]
        ]

        train_dataset = PACSDataset(
            train_samples,
            transform=get_transforms(train=True)
        )

        val_dataset = PACSDataset(
            val_samples,
            transform=get_transforms(train=False)
        )

        train_loaders[domain] = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE_PER_DOMAIN,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )

        val_loaders[domain] = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )

        print(
            f"{domain}: "
            f"{len(train_dataset)} train, "
            f"{len(val_dataset)} validation"
        )

    return train_loaders, val_loaders


# ============================================================
# Source validation
# ============================================================

def validate_sources(
    backbone,
    classifier,
    val_loaders
):

    source_results = {}

    for domain in SOURCE_DOMAINS:

        accuracy, macro_f1 = evaluate(
            backbone,
            classifier,
            val_loaders[domain]
        )

        source_results[domain] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1
        }

    mean_source_f1 = np.mean(
        [
            source_results[d]["macro_f1"]
            for d in SOURCE_DOMAINS
        ]
    )

    return source_results, mean_source_f1


# ============================================================
# Print source validation
# ============================================================

def print_source_results(
    source_results,
    mean_source_f1
):

    print(
        f"Mean Source Macro-F1: "
        f"{mean_source_f1:.4f}"
    )

    for domain in SOURCE_DOMAINS:

        print(
            f"  {domain:13s} | "
            f"Acc: "
            f"{source_results[domain]['accuracy']:.4f} | "
            f"Macro-F1: "
            f"{source_results[domain]['macro_f1']:.4f}"
        )


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path,
    epoch,
    backbone,
    classifier,
    optimizer,
    best_mean_f1
):

    torch.save(
        {
            "epoch": epoch,
            "backbone": backbone.state_dict(),
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_mean_source_macro_f1": best_mean_f1,
            "seed": SEED
        },
        path
    )

    print(
        f"  --> Best checkpoint saved: {path}"
    )


# ============================================================
# Generic source-only training loop
# ============================================================

def train_method(
    method,
    source_loader,
    val_loaders,
    checkpoint_path
):

    backbone = method.backbone
    classifier = method.classifier
    optimizer = method.optimizer

    best_mean_f1 = -float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):

        method.train()

        if hasattr(
            backbone,
            "freeze_batchnorm_stats"
        ):
            backbone.freeze_batchnorm_stats()

        running_loss = 0.0
        num_batches = 0

        for images, labels, domains in source_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            loss = method.compute_loss(
                images,
                labels,
                domains
            )

            loss.backward()
            optimizer.step()

            if hasattr(
                backbone,
                "freeze_batchnorm_stats"
            ):
                backbone.freeze_batchnorm_stats()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches

        source_results, mean_source_f1 = validate_sources(
            backbone,
            classifier,
            val_loaders
        )

        print(
            f"\nEpoch {epoch:02d} | "
            f"Loss: {avg_loss:.4f} | "
            f"Mean Source Macro-F1: "
            f"{mean_source_f1:.4f}"
        )

        print_source_results(
            source_results,
            mean_source_f1
        )

        if mean_source_f1 > best_mean_f1:

            best_mean_f1 = mean_source_f1
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "backbone":
                        backbone.state_dict(),
                    "classifier":
                        classifier.state_dict(),
                    "optimizer":
                        optimizer.state_dict(),
                    "best_mean_source_macro_f1":
                        best_mean_f1,
                    "seed": SEED
                },
                checkpoint_path
            )

            print(
                f"  --> Best checkpoint saved: "
                f"{checkpoint_path}"
            )

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:

            print(
                f"\nEarly stopping at epoch "
                f"{epoch}."
            )

            break

    return best_mean_f1


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    with open(
        args.config,
        "r"
    ) as f:

        config = yaml.safe_load(f)

    seed = config.get(
        "seed",
        SEED
    )

    set_seed(seed)

    method_name = config.get(
        "method"
    )

    split_file = config["data"]["split_file"]

    checkpoint_dir = Path(
        config["output"]["checkpoint_dir"]
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load Task 2 split
    # --------------------------------------------------------

    with open(
        split_file,
        "r"
    ) as f:

        split_data = json.load(f)

    # --------------------------------------------------------
    # IMPORTANT:
    # Only source domains are loaded.
    #
    # There is NO target/sketch loader here.
    # --------------------------------------------------------

    (
        train_loaders,
        val_loaders
    ) = build_source_loaders(
        split_data
    )

    source_loader = CyclingDomainLoader(
        train_loaders
    )

    print("\nDevice:", DEVICE)
    print("Method:", method_name)

    # ========================================================
    # ERM
    # ========================================================

    if method_name == "erm":

        backbone = ResNet18Backbone().to(
            DEVICE
        )

        classifier = ClassifierHead(
            backbone.feature_dim,
            NUM_CLASSES
        ).to(DEVICE)

        # ----------------------------------------------------
        # IMPORTANT:
        # Load Task 2 Source-only ERM checkpoint.
        # Do NOT retrain ERM.
        # ----------------------------------------------------

        erm_checkpoint = Path(
            config["erm_checkpoint"]
        )

        checkpoint = torch.load(
            erm_checkpoint,
            map_location=DEVICE
        )

        backbone.load_state_dict(
            checkpoint["backbone"]
        )

        classifier.load_state_dict(
            checkpoint["classifier"]
        )

        print(
            "\nLoaded Task 2 ERM checkpoint:"
        )

        print(
            erm_checkpoint
        )

        print(
            "\nERM baseline loaded. "
            "No ERM retraining performed."
        )

        return

    # ========================================================
    # DAN-DG
    # ========================================================

    elif method_name == "dan_dg":

        model = DANDGMethod(
            num_classes=NUM_CLASSES,
            lambda_mmd=config[
                "adaptation"
            ].get(
                "lambda",
                1.0
            )
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        checkpoint_path = (
            checkpoint_dir /
            "dan_dg_best.pth"
        )

        best_f1 = train_method(
            model,
            model.backbone,
            model.classifier,
            optimizer,
            source_loader,
            val_loaders,
            checkpoint_path
        )

    # ========================================================
    # SAM
    # ========================================================

    elif method_name == "sam":

        model = SAMMethod(
            num_classes=NUM_CLASSES,
            **config.get(
                "sam",
                {}
            )
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        checkpoint_path = (
            checkpoint_dir /
            "sam_best.pth"
        )

        best_f1 = train_method(
            model,
            model.backbone,
            model.classifier,
            optimizer,
            source_loader,
            val_loaders,
            checkpoint_path
        )

    else:

        raise ValueError(
            f"Unsupported method: "
            f"{method_name}"
        )

    print(
        "\nTraining complete."
    )

    print(
        "Best mean source validation "
        "Macro-F1:",
        best_f1
    )

    print(
        "Checkpoint:",
        checkpoint_path
    )


if __name__ == "__main__":
    main()