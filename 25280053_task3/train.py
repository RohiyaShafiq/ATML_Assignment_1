# ============================================================
# Task 3 - Training
# PACS Domain Generalization
#
# Sources: Photo, Art Painting, Cartoon
# Target: Sketch (NEVER loaded here)
# Seed: 6304
# ============================================================

import os
import json
import random
import numpy as np
import torch
import sys

sys.path.append("/content/drive/MyDrive")

from torch.utils.data import DataLoader

from shared.pacs import PACSDataset, get_transforms

from task3.models.backbone import ResNet18Backbone
from task3.models.classifier_head import ClassifierHead

from task3.methods.erm import SourceOnlyMethod
from task3.methods.dan_dg import DANMethod
from task3.methods.sam import SAMMethod

from task3.evaluation.domain_metrics import (
    evaluate_source_domains
)


# ============================================================
# Configuration
# ============================================================

SEED = 6304

DATA_ROOT = "/content/drive/MyDrive/PACS/PACS"
SPLIT_FILE = "/content/drive/MyDrive/shared/splits/pacs_sketch_seed6304.json"

RESULTS_DIR = "/content/drive/MyDrive/task3/results"

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

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Load split file
# ============================================================

def load_split_file():

    with open(SPLIT_FILE, "r") as f:
        split_data = json.load(f)

    return split_data


# ============================================================
# Macro F1
# ============================================================

def macro_f1_from_confusion_matrix(cm):

    f1_scores = []

    for c in range(NUM_CLASSES):

        tp = cm[c, c]

        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

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
# Evaluate one source domain
# ============================================================

@torch.no_grad()
def evaluate(backbone, classifier, loader):

    backbone.eval()
    classifier.eval()

    confusion_matrix = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64
    )

    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        features = backbone(images)
        logits = classifier(features)

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        for true_label, pred_label in zip(
            labels.cpu().numpy(),
            predictions.cpu().numpy()
        ):
            confusion_matrix[
                true_label,
                pred_label
            ] += 1

    accuracy = correct / total

    macro_f1 = macro_f1_from_confusion_matrix(
        confusion_matrix
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion_matrix
    }


# ============================================================
# Source validation
# ============================================================

@torch.no_grad()
def evaluate_sources(
    backbone,
    classifier,
    source_val_loaders
):

    results = {}

    macro_f1_values = []

    for domain in SOURCE_DOMAINS:

        metrics = evaluate(
            backbone,
            classifier,
            source_val_loaders[domain]
        )

        results[domain] = metrics

        macro_f1_values.append(
            metrics["macro_f1"]
        )

    mean_macro_f1 = np.mean(
        macro_f1_values
    )

    results["mean_macro_f1"] = float(
        mean_macro_f1
    )

    return results


# ============================================================
# Domain-balanced source loader
# ============================================================

class CyclingDomainLoader:

    def __init__(
        self,
        domain_loaders
    ):

        self.domain_loaders = domain_loaders

        self.iterators = {
            domain: iter(loader)
            for domain, loader
            in domain_loaders.items()
        }

    def __iter__(self):

        return self

    def __next__(self):

        images = []
        labels = []
        domains = []

        for domain in SOURCE_DOMAINS:

            try:
                x, y = next(
                    self.iterators[domain]
                )

            except StopIteration:

                self.iterators[domain] = iter(
                    self.domain_loaders[domain]
                )

                x, y = next(
                    self.iterators[domain]
                )

            images.append(x)
            labels.append(y)

            domains.extend(
                [domain] * len(y)
            )

        images = torch.cat(
            images,
            dim=0
        )

        labels = torch.cat(
            labels,
            dim=0
        )

        return images, labels, domains


# ============================================================
# Build source datasets
# ============================================================

def build_datasets():

    with open(SPLIT_FILE, "r") as f:
        split_data = json.load(f)

    train_datasets = {}
    val_datasets = {}

    for domain in SOURCE_DOMAINS:

        train_samples = split_data["sources"][domain]["train"]
        val_samples = split_data["sources"][domain]["val"]

        train_datasets[domain] = PACSDataset(
            train_samples,
            transform=get_transforms(train=True)
        )

        val_datasets[domain] = PACSDataset(
            val_samples,
            transform=get_transforms(train=False)
        )

    return train_datasets, val_datasets 


# ============================================================
# Build source loaders
# ============================================================

def build_loaders(
    train_datasets,
    val_datasets
):

    train_loaders = {}
    val_loaders = {}

    for domain in SOURCE_DOMAINS:

        train_loaders[domain] = DataLoader(
            train_datasets[domain],
            batch_size=BATCH_SIZE_PER_DOMAIN,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        val_loaders[domain] = DataLoader(
            val_datasets[domain],
            batch_size=64,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

    return train_loaders, val_loaders


# ============================================================
# Build model
# ============================================================

def build_model():

    backbone = ResNet18Backbone().to(DEVICE)

    classifier = ClassifierHead(
        backbone.feature_dim,
        NUM_CLASSES
    ).to(DEVICE)

    return backbone, classifier
# ============================================================
# Load Task 2 ERM checkpoint
# ============================================================

def load_task2_erm_checkpoint(
    backbone,
    classifier
):

    checkpoint_path = (
        "/content/drive/MyDrive/"
        "task2/results/"
        "source_only_erm_best.pth"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    backbone.load_state_dict(
        checkpoint["backbone"]
    )

    classifier.load_state_dict(
        checkpoint["classifier"]
    )

    print(
        f"Loaded Task 2 ERM checkpoint:\n"
        f"{checkpoint_path}"
    )

    return backbone, classifier


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(
    path,
    backbone,
    classifier,
    epoch,
    mean_macro_f1
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    torch.save(
        {
            "backbone": backbone.state_dict(),
            "classifier": classifier.state_dict(),
            "epoch": epoch,
            "mean_source_macro_f1": mean_macro_f1,
            "seed": SEED
        },
        path
    )


# ============================================================
# Train one method
# ============================================================

def train_method(
    method_name,
    method,
    backbone,
    classifier,
    train_loaders,
    val_loaders
):

    best_macro_f1 = -float("inf")

    best_epoch = 0

    epochs_without_improvement = 0

    checkpoint_path = os.path.join(
        RESULTS_DIR,
        f"{method_name}_best.pth"
    )

    # Number of iterations per epoch.
    # Each iteration contains 8 samples
    # from every source domain.
    steps_per_epoch = max(
        len(loader)
        for loader in train_loaders.values()
    )

    train_loader = CyclingDomainLoader(
        train_loaders
    )

    for epoch in range(1, EPOCHS + 1):

        backbone.train()
        classifier.train()

        for step in range(
            steps_per_epoch
        ):

            images, labels, domains = next(
                train_loader
            )

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # ------------------------------------------------
            # Method-specific training step
            # ------------------------------------------------

            loss = method.training_step(
                images=images,
                labels=labels,
                domains=domains
            )

        # ----------------------------------------------------
        # Source-only validation
        # ----------------------------------------------------

        validation_results = evaluate_sources(
            backbone,
            classifier,
            val_loaders
        )

        mean_macro_f1 = (
            validation_results[
                "mean_macro_f1"
            ]
        )

        print(
            f"\n{method_name} | "
            f"Epoch {epoch}/{EPOCHS} | "
            f"Mean Source Macro-F1: "
            f"{mean_macro_f1:.4f}"
        )

        for domain in SOURCE_DOMAINS:

            print(
                f"  {domain}: "
                f"Acc={validation_results[domain]['accuracy']:.4f}, "
                f"F1={validation_results[domain]['macro_f1']:.4f}"
            )

        # ----------------------------------------------------
        # Checkpoint selection
        # ----------------------------------------------------

        if mean_macro_f1 > best_macro_f1:

            best_macro_f1 = mean_macro_f1

            best_epoch = epoch

            epochs_without_improvement = 0

            save_checkpoint(
                checkpoint_path,
                backbone,
                classifier,
                epoch,
                mean_macro_f1
            )

            print(
                f"  Saved best checkpoint."
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= PATIENCE:

            print(
                f"\nEarly stopping at epoch "
                f"{epoch}."
            )

            break

    print(
        f"\nBest {method_name} checkpoint:"
    )

    print(
        f"Epoch: {best_epoch}"
    )

    print(
        f"Mean source Macro-F1: "
        f"{best_macro_f1:.4f}"
    )

    return checkpoint_path

# ============================================================
# Part 2 - Train DAN-DG
# ============================================================

def train_dan_dg(
    train_loaders,
    val_loaders
):

    print("\n" + "=" * 70)
    print("PART 2 - DAN-DG")

    # --------------------------------------------------------
    # Build DAN-DG model
    # --------------------------------------------------------

    method = DANMethod(
        num_classes=NUM_CLASSES,
        lambda_dg=1.0
    ).to(DEVICE)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        method.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_macro_f1 = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    checkpoint_path = os.path.join(
        RESULTS_DIR,
        "dan_dg_best.pth"
    )

    # --------------------------------------------------------
    # Number of batches per epoch
    # --------------------------------------------------------

    steps_per_epoch = max(
        len(train_loaders[domain])
        for domain in SOURCE_DOMAINS
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        method.train()

        # Create fresh iterators every epoch
        iterators = {
            domain: iter(train_loaders[domain])
            for domain in SOURCE_DOMAINS
        }

        total_loss = 0.0
        total_erm = 0.0
        total_mmd = 0.0

        for step in range(steps_per_epoch):

            source_batches = {}

            # ------------------------------------------------
            # Get one batch from each source domain
            # Each batch contains 8 samples
            # ------------------------------------------------

            for domain in SOURCE_DOMAINS:

                try:

                    images, labels, _ = next(
                        iterators[domain]
                    )

                except StopIteration:

                    iterators[domain] = iter(
                        train_loaders[domain]
                    )

                    images, labels, _ = next(
                        iterators[domain]
                    )

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                source_batches[domain] = (
                    images,
                    labels
                )

            # ------------------------------------------------
            # DAN-DG loss
            #
            # L = L_ERM + lambda_DG * average MMD
            # ------------------------------------------------

            optimizer.zero_grad()

            loss, erm_loss, mmd_loss = method.loss(
                source_batches
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            total_erm += erm_loss.item()
            total_mmd += mmd_loss.item()

        # ----------------------------------------------------
        # Average training losses
        # ----------------------------------------------------

        avg_loss = total_loss / steps_per_epoch
        avg_erm = total_erm / steps_per_epoch
        avg_mmd = total_mmd / steps_per_epoch

        # ----------------------------------------------------
        # Source validation
        # ----------------------------------------------------

        validation_results = evaluate_source_domains(
            method.backbone,
            method.classifier,
            val_loaders,
            DEVICE
        )

        mean_macro_f1 = (
            validation_results["mean"]["macro_f1"]
        )

        print(
            f"\nDAN-DG | "
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"  Train Loss: {avg_loss:.4f} | "
            f"ERM: {avg_erm:.4f} | "
            f"MMD: {avg_mmd:.4f}"
        )

        print(
            f"  Mean Source Macro-F1: "
            f"{mean_macro_f1:.4f}"
        )

        for domain in SOURCE_DOMAINS:

            print(
                f"  {domain}: "
                f"Acc="
                f"{validation_results[domain]['accuracy']:.4f}, "
                f"F1="
                f"{validation_results[domain]['macro_f1']:.4f}"
            )

        # ----------------------------------------------------
        # Checkpoint selection
        # ----------------------------------------------------

        if mean_macro_f1 > best_macro_f1:

            best_macro_f1 = mean_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                checkpoint_path,
                method.backbone,
                method.classifier,
                epoch,
                mean_macro_f1
            )

            print(
                "  Saved best DAN-DG checkpoint."
            )

        else:

            epochs_without_improvement += 1

            print(
                f"  No improvement "
                f"({epochs_without_improvement}/"
                f"{PATIENCE})"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if epochs_without_improvement >= PATIENCE:

            print(
                f"\nEarly stopping at epoch "
                f"{epoch}."
            )

            break

    # --------------------------------------------------------
    # Final training information
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DAN-DG TRAINING COMPLETE")

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best mean source Macro-F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    return checkpoint_path

# ============================================================
# Part 3 - SAM training
# ============================================================

def train_sam(train_loaders, val_loaders, rho):
    print("=" * 70)
    print(f"PART 5 - SAM SENSITIVITY, rho={rho}")

    method = SAMMethod(
        num_classes=NUM_CLASSES,
        rho=rho
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        method.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_macro_f1 = -float("inf")
    best_checkpoint = None
    patience_counter = 0

    steps_per_epoch = max(
        len(train_loaders[domain])
        for domain in SOURCE_DOMAINS
    )

    for epoch in range(1, EPOCHS + 1):

        method.train()

        iterators = {
            domain: iter(train_loaders[domain])
            for domain in SOURCE_DOMAINS
        }

        for step in range(steps_per_epoch):

            all_images = []
            all_labels = []

            for domain in SOURCE_DOMAINS:

                try:
                    images, labels, _ = next(
                        iterators[domain]
                    )
                except StopIteration:
                    iterators[domain] = iter(
                        train_loaders[domain]
                    )

                    images, labels, _ = next(
                        iterators[domain]
                    )

                all_images.append(images)
                all_labels.append(labels)

            images = torch.cat(all_images, dim=0).to(DEVICE)
            labels = torch.cat(all_labels, dim=0).to(DEVICE)

            # ------------------------------------------------
            # First SAM pass
            # ------------------------------------------------

            optimizer.zero_grad()

            method.freeze_bn_stats()

            loss = method.loss(
                images,
                labels
            )

            loss.backward()

            grad_norm = torch.norm(
                torch.stack([
                    p.grad.norm(2)
                    for p in method.parameters()
                    if p.grad is not None
                ]),
                2
            )

            scale = rho / (
                grad_norm + 1e-12
            )

            perturbations = []

            with torch.no_grad():

                for parameter in method.parameters():

                    if parameter.grad is None:
                        perturbations.append(None)
                        continue

                    epsilon = (
                        parameter.grad * scale
                    )

                    parameter.add_(epsilon)
                    perturbations.append(epsilon)

            # ------------------------------------------------
            # Second SAM pass
            # ------------------------------------------------

            optimizer.zero_grad()

            method.freeze_bn_stats()

            perturbed_loss = method.loss(
                images,
                labels
            )

            perturbed_loss.backward()

            # ------------------------------------------------
            # Restore original parameters
            # ------------------------------------------------

            with torch.no_grad():

                for parameter, epsilon in zip(
                    method.parameters(),
                    perturbations
                ):

                    if epsilon is not None:
                        parameter.sub_(epsilon)

            optimizer.step()

        # ----------------------------------------------------
        # Source validation
        # ----------------------------------------------------

        validation_results = evaluate_source_domains(
            method.backbone,
            method.classifier,
            val_loaders,
            DEVICE
        )

        mean_macro_f1 = (
            validation_results["mean"]["macro_f1"]
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Mean Source Macro-F1: "
            f"{mean_macro_f1:.4f}"
        )

        if mean_macro_f1 > best_macro_f1:

            best_macro_f1 = mean_macro_f1
            patience_counter = 0

            checkpoint_path = (
                f"/content/drive/MyDrive/task3/results/"
                f"sam_rho_{rho}.pth"
            )

            torch.save(
                {
                    "backbone": method.backbone.state_dict(),
                    "classifier": method.classifier.state_dict(),
                    "rho": rho,
                    "best_macro_f1": best_macro_f1
                },
                checkpoint_path
            )

            best_checkpoint = checkpoint_path

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    f"Early stopping at epoch {epoch}"
                )

                break

    return best_checkpoint

# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    # --------------------------------------------------------
    # Build source datasets
    # --------------------------------------------------------
    train_datasets, val_datasets = build_datasets()

    # --------------------------------------------------------
    # Build source loaders
    # --------------------------------------------------------
    train_loaders, val_loaders = build_loaders(
        train_datasets,
        val_datasets
    )

    # ========================================================
    # Part 1 - Source-only ERM baseline
    # ========================================================

    TASK2_ERM_CHECKPOINT = ("/content/drive/MyDrive/source_only_erm_best.pth")

    backbone, classifier = build_model()

    checkpoint = torch.load(
        TASK2_ERM_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    backbone.load_state_dict(
        checkpoint["backbone"]
    )

    classifier.load_state_dict(
        checkpoint["classifier"]
    )

    # IMPORTANT:
    # If your val_loaders use keys like
    # "photo_val", "art_painting_val", "cartoon_val",
    # create the mapping here.
    source_val_loaders = {
        "photo": val_loaders["photo"],
        "art_painting": val_loaders["art_painting"],
        "cartoon": val_loaders["cartoon"],
    }

    source_results = evaluate_source_domains(
        backbone,
        classifier,
        source_val_loaders,
        DEVICE
    )

    print("\n" + "=" * 70)
    print("SOURCE-ONLY ERM BASELINE")

    for domain in SOURCE_DOMAINS:
        print(
            f"{domain:15s} | "
            f"Accuracy: "
            f"{source_results[domain]['accuracy']:.4f} | "
            f"Macro-F1: "
            f"{source_results[domain]['macro_f1']:.4f}"
        )

    print(
        f"\nMean             | "
        f"Accuracy: "
        f"{source_results['mean']['accuracy']:.4f} | "
        f"Macro-F1: "
        f"{source_results['mean']['macro_f1']:.4f}"
    )

    print(
        f"Worst-domain     | "
        f"Accuracy: "
        f"{source_results['worst_domain']['accuracy']:.4f} | "
        f"Macro-F1: "
        f"{source_results['worst_domain']['macro_f1']:.4f}"
    )

    dan_checkpoint = train_dan_dg(
        train_loaders,
        val_loaders
    )

    sam_checkpoint = train_sam(
        train_loaders,
        val_loaders
    )


if __name__ == "__main__":

    set_seed(SEED)

    train_datasets, val_datasets = build_datasets()

    train_loaders, val_loaders = build_loaders(
        train_datasets,
        val_datasets
    )

    for rho in [0.01, 0.05, 0.1]:

        set_seed(SEED)

        train_sam(
            train_loaders,
            val_loaders,
            rho
        )