# ============================================================
# Task 3 - Part 4 Evaluation
# PACS Domain Generalization
#
# ERM vs DAN-DG vs SAM
#
# Sketch is loaded ONLY here.
# ============================================================

import os
import json
import random

import numpy as np
import torch

import sys

sys.path.insert(
    0,
    "/content/drive/MyDrive"
)

from torch.utils.data import DataLoader

from shared.pacs import (
    PACSDataset,
    get_transforms
)

from task3.models.backbone import (
    ResNet18Backbone
)

from task3.models.classifier_head import (
    ClassifierHead
)

from task3.evaluation.final_metrics import (
    evaluate_source_domains,
    evaluate_domain
)

from task3.evaluation.source_domain_separability import (
    source_domain_separability
)

from task3.evaluation.sharpness import (
    collect_fixed_validation_batch,
    local_sharpness
)


# ============================================================
# Configuration
# ============================================================

SEED = 6304

SPLIT_FILE = (
    "/content/drive/MyDrive/shared/splits/"
    "pacs_sketch_seed6304.json"
)

RESULTS_DIR = (
    "/content/drive/MyDrive/task3/results"
)

SOURCE_DOMAINS = [
    "photo",
    "art_painting",
    "cartoon"
]

NUM_CLASSES = 7

BATCH_SIZE = 64

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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
# Build model
# ============================================================

def build_model():

    backbone = ResNet18Backbone().to(
        DEVICE
    )

    classifier = ClassifierHead(
        backbone.feature_dim,
        NUM_CLASSES
    ).to(DEVICE)

    return backbone, classifier


# ============================================================
# Load checkpoint
# ============================================================

def load_checkpoint(
    checkpoint_path
):

    backbone, classifier = build_model()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    backbone.load_state_dict(
        checkpoint["backbone"]
    )

    classifier.load_state_dict(
        checkpoint["classifier"]
    )

    backbone.eval()
    classifier.eval()

    return backbone, classifier


# ============================================================
# Build source validation datasets
# ============================================================

def build_source_validation_loaders():

    with open(
        SPLIT_FILE,
        "r"
    ) as f:

        split_data = json.load(f)

    val_loaders = {}

    for domain in SOURCE_DOMAINS:

        samples = (
            split_data[
                "sources"
            ][domain]["val"]
        )

        dataset = PACSDataset(
            samples,
            transform=get_transforms(
                train=False
            )
        )

        val_loaders[domain] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

    return val_loaders


# ============================================================
# Build Sketch test loader
# ============================================================

def build_sketch_loader():
    with open(SPLIT_FILE, "r") as f:
        split_data = json.load(f)

    sketch_samples = split_data["target"]["adaptation"]

    sketch_dataset = PACSDataset(
        sketch_samples,
        transform=get_transforms(train=False)
    )

    sketch_loader = DataLoader(
        sketch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return sketch_loader


# ============================================================
# Evaluate one model
# ============================================================

def evaluate_model(
    model_name,
    checkpoint_path,
    source_val_loaders,
    sketch_loader,
    fixed_sharpness_batch
):

    print("\n" + "=" * 70)
    print(model_name)

    backbone, classifier = load_checkpoint(
        checkpoint_path
    )

    # ========================================================
    # Source validation
    # ========================================================

    source_results = evaluate_source_domains(
        backbone,
        classifier,
        source_val_loaders,
        DEVICE
    )

    print("\nSource Validation:")

    for domain in SOURCE_DOMAINS:

        print(
            f"{domain:15s} | "
            f"Accuracy: "
            f"{source_results[domain]['accuracy']:.4f} | "
            f"Macro-F1: "
            f"{source_results[domain]['macro_f1']:.4f}"
        )

    print(
        f"\nMean           | "
        f"Accuracy: "
        f"{source_results['mean']['accuracy']:.4f} | "
        f"Macro-F1: "
        f"{source_results['mean']['macro_f1']:.4f}"
    )

    print(
        f"Worst-domain   | "
        f"Accuracy: "
        f"{source_results['worst_domain']['accuracy']:.4f} | "
        f"Macro-F1: "
        f"{source_results['worst_domain']['macro_f1']:.4f}"
    )

    # ========================================================
    # Sketch evaluation
    # ========================================================

    sketch_results = evaluate_domain(
        backbone,
        classifier,
        sketch_loader,
        DEVICE
    )

    print("\nSketch:")

    print(
        f"Accuracy: "
        f"{sketch_results['accuracy']:.4f}"
    )

    print(
        f"Macro-F1: "
        f"{sketch_results['macro_f1']:.4f}"
    )

    # ========================================================
    # Source-domain separability
    # ========================================================

    separability = (
        source_domain_separability(
            backbone,
            source_val_loaders,
            DEVICE,
            seed=SEED
        )
    )

    print(
        "\nSource-domain separability: "
        f"{separability['accuracy']:.4f}"
    )

    print(
        "Chance performance: "
        f"{separability['chance']:.4f}"
    )

    # ========================================================
    # Sharpness
    # ========================================================

    images, labels = (
        fixed_sharpness_batch
    )

    sharpness = local_sharpness(
        backbone,
        classifier,
        images,
        labels,
        rho=0.05
    )

    print(
        "\nSharpness:"
    )

    print(
        f"Original loss: "
        f"{sharpness['original_loss']:.6f}"
    )

    print(
        f"Perturbed loss: "
        f"{sharpness['perturbed_loss']:.6f}"
    )

    print(
        f"Delta sharp: "
        f"{sharpness['delta_sharp']:.6f}"
    )

    return {
        "source": source_results,
        "sketch": sketch_results,
        "separability": separability,
        "sharpness": sharpness
    }


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    # --------------------------------------------------------
    # Source validation loaders
    # --------------------------------------------------------

    source_val_loaders = (
        build_source_validation_loaders()
    )

    # --------------------------------------------------------
    # Sketch is loaded ONLY now, after all models are fixed.
    # --------------------------------------------------------

    sketch_loader = build_sketch_loader()

    # --------------------------------------------------------
    # Same fixed 32-per-domain batch for all models
    # --------------------------------------------------------

    fixed_sharpness_batch = (
        collect_fixed_validation_batch(
            source_val_loaders,
            DEVICE,
            seed=SEED,
            samples_per_domain=32
        )
    )

    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    checkpoints = {

        "ERM": os.path.join(
            "/content/drive/MyDrive/source_only_erm_best.pth"
        ),

        "DAN-DG": os.path.join(
            RESULTS_DIR,
            "dan_dg_best.pth"
        ),

        "SAM": os.path.join(
            RESULTS_DIR,
            "sam_best.pth"
        )
    }

    results = {}

    # ========================================================
    # Evaluate all three models
    # ========================================================

    for model_name, checkpoint_path in checkpoints.items():

        if not os.path.exists(
            checkpoint_path
        ):

            print(
                f"\nWARNING: checkpoint not found:"
                f"\n{checkpoint_path}"
            )

            continue

        results[model_name] = evaluate_model(
            model_name,
            checkpoint_path,
            source_val_loaders,
            sketch_loader,
            fixed_sharpness_batch
        )

    # ========================================================
    # Sketch accuracy change relative to ERM
    # ========================================================

    if "ERM" in results:

        erm_accuracy = (
            results["ERM"]
            ["sketch"]
            ["accuracy"]
        )

        print("\n" + "=" * 70)
        print("SKETCH ACCURACY CHANGE RELATIVE TO ERM")

        for model_name in [
            "DAN-DG",
            "SAM"
        ]:

            if model_name not in results:
                continue

            model_accuracy = (
                results[model_name]
                ["sketch"]
                ["accuracy"]
            )

            change = (
                model_accuracy
                - erm_accuracy
            )

            print(
                f"{model_name:10s}: "
                f"{change:+.4f}"
            )

    # ========================================================
    # Final comparison table
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")

    print(
        f"{'Model':10s} "
        f"{'Src Acc':>10s} "
        f"{'Src F1':>10s} "
        f"{'Sketch Acc':>12s} "
        f"{'Sketch F1':>12s} "
        f"{'Domain Sep':>12s} "
        f"{'Sharpness':>12s}"
    )

    for model_name, result in results.items():

        print(
            f"{model_name:10s} "
            f"{result['source']['mean']['accuracy']:10.4f} "
            f"{result['source']['mean']['macro_f1']:10.4f} "
            f"{result['sketch']['accuracy']:12.4f} "
            f"{result['sketch']['macro_f1']:12.4f} "
            f"{result['separability']['accuracy']:12.4f} "
            f"{result['sharpness']['delta_sharp']:12.6f}"
        )

    # ========================================================
    # Per-class Sketch analysis
    # ========================================================

    if results:

        print("\n" + "=" * 70)
        print("PER-CLASS SKETCH CONFUSION MATRICES")

        for model_name, result in results.items():

            print(
                f"\n{model_name}"
            )

            print(
                result["sketch"]
                ["confusion_matrix"]
            )


if __name__ == "__main__":
    main()