import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from shared.pacs import PACSDataset, get_transforms
from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import ClassifierHead


# ============================================================
# Configuration
# ============================================================

SEED = 6304

DATA_ROOT = Path(
    "/content/drive/MyDrive/PACS/PACS"
)

SPLIT_FILE = Path(
    "/content/drive/MyDrive/shared/splits/"
    "pacs_sketch_seed6304.json"
)

RESULTS_DIR = Path(
    "/content/drive/MyDrive/task2/results"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SOURCE_DOMAINS = [
    "photo",
    "art_painting",
    "cartoon",
]

TARGET_DOMAIN = "sketch"

CLASSES = [
    "dog",
    "elephant",
    "giraffe",
    "guitar",
    "horse",
    "house",
    "person",
]

NUM_CLASSES = 7

BATCH_SIZE = 64


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Load split file
# ============================================================

def load_split_file():

    with open(SPLIT_FILE, "r") as f:
        return json.load(f)


# ============================================================
# Build datasets
# ============================================================

def build_datasets(split_data):

    datasets = {}

    # --------------------------------------------------------
    # Source validation
    # --------------------------------------------------------

    for domain in SOURCE_DOMAINS:

        samples = [
            tuple(sample)
            for sample in split_data["sources"][domain]["val"]
        ]

        datasets[f"{domain}_val"] = PACSDataset(
            samples,
            transform=get_transforms(train=False)
        )

    # --------------------------------------------------------
    # Target adaptation / target final set
    # --------------------------------------------------------

    target_samples = [
        tuple(sample)
        for sample in split_data["target"]["adaptation"]
    ]

    datasets["target"] = PACSDataset(
        target_samples,
        transform=get_transforms(train=False)
    )

    return datasets


# ============================================================
# DataLoaders
# ============================================================

def build_loaders(datasets):

    loaders = {}

    for name, dataset in datasets.items():

        loaders[name] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )

    return loaders


# ============================================================
# Load checkpoint
# ============================================================

def load_model(checkpoint_path):

    backbone = ResNet18Backbone().to(DEVICE)

    classifier = ClassifierHead(
        backbone.feature_dim,
        NUM_CLASSES
    ).to(DEVICE)

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
# Classification evaluation
# ============================================================

@torch.no_grad()
def evaluate_classifier(
    backbone,
    classifier,
    loader
):

    all_predictions = []
    all_labels = []

    backbone.eval()
    classifier.eval()

    for images, labels, _ in loader:

        images = images.to(DEVICE)

        features = backbone(images)

        logits = classifier(features)

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return (
        accuracy,
        macro_f1,
        np.array(all_labels),
        np.array(all_predictions)
    )


# ============================================================
# Evaluate source validation + target
# ============================================================

def evaluate_method(
    method_name,
    checkpoint_path,
    loaders
):


    print("\n" + "=" * 70)
    print(f"{method_name}")

    backbone, classifier = load_model(
        checkpoint_path
    )

    source_results = {}

    for domain in SOURCE_DOMAINS:

        (
            accuracy,
            macro_f1,
            _,
            _
        ) = evaluate_classifier(
            backbone,
            classifier,
            loaders[f"{domain}_val"]
        )

        source_results[domain] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1
        }

        print(
            f"{domain:13s} | "
            f"Acc: {accuracy:.4f} | "
            f"Macro-F1: {macro_f1:.4f}"
        )

    # --------------------------------------------------------
    # Mean source validation
    # --------------------------------------------------------

    mean_source_accuracy = np.mean(
        [
            source_results[d]["accuracy"]
            for d in SOURCE_DOMAINS
        ]
    )

    mean_source_f1 = np.mean(
        [
            source_results[d]["macro_f1"]
            for d in SOURCE_DOMAINS
        ]
    )

    print(
        f"\nMean Source Val | "
        f"Acc: {mean_source_accuracy:.4f} | "
        f"Macro-F1: {mean_source_f1:.4f}"
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    (
        target_accuracy,
        target_macro_f1,
        target_labels,
        target_predictions
    ) = evaluate_classifier(
        backbone,
        classifier,
        loaders["target"]
    )

    print(
        f"Target          | "
        f"Acc: {target_accuracy:.4f} | "
        f"Macro-F1: {target_macro_f1:.4f}"
    )

    return {
        "method": method_name,
        "checkpoint": str(checkpoint_path),
        "backbone": backbone,
        "classifier": classifier,
        "source_results": source_results,
        "mean_source_accuracy": mean_source_accuracy,
        "mean_source_macro_f1": mean_source_f1,
        "target_accuracy": target_accuracy,
        "target_macro_f1": target_macro_f1,
        "target_labels": target_labels,
        "target_predictions": target_predictions,
    }


# Feature extraction
@torch.no_grad()
def collect_features(
    backbone,
    loader,
    max_samples=None
):

    backbone.eval()

    features = []
    labels = []

    collected = 0

    for images, batch_labels, _ in loader:

        images = images.to(DEVICE)

        batch_features = backbone(images)

        features.append(
            batch_features.cpu().numpy()
        )

        labels.append(
            batch_labels.numpy()
        )

        collected += len(batch_labels)

        if (
            max_samples is not None
            and collected >= max_samples
        ):
            break

    features = np.concatenate(
        features,
        axis=0
    )

    labels = np.concatenate(
        labels,
        axis=0
    )

    if max_samples is not None:

        features = features[:max_samples]
        labels = labels[:max_samples]

    return features, labels


# Domain separability
def domain_separability(
    backbone,
    source_loaders,
    target_loader
):

    # Collect equal number of source and target features
    source_counts = [
        len(source_loaders[d].dataset)
        for d in SOURCE_DOMAINS
    ]

    target_count = len(
        target_loader.dataset
    )

    # Equal number of source-validation and target features
    n_each = min(
        min(source_counts),
        target_count
    )

    # Use the same number from each source domain
    source_features = []

    for domain in SOURCE_DOMAINS:

        features, _ = collect_features(
            backbone,
            source_loaders[domain],
            max_samples=n_each
        )

        source_features.append(
            features
        )

    source_features = np.concatenate(
        source_features,
        axis=0
    )

    # Equal number of source and target examples
    # Source total is 3 * n_each, so select the same number of target features.
    total_source = len(source_features)

    target_features, _ = collect_features(
        backbone,
        target_loader,
        max_samples=total_source
    )

    n = min(
        len(source_features),
        len(target_features)
    )

    source_features = source_features[:n]
    target_features = target_features[:n]

    X = np.concatenate(
        [
            source_features,
            target_features
        ],
        axis=0
    )

    y = np.concatenate(
        [
            np.zeros(n, dtype=np.int64),
            np.ones(n, dtype=np.int64)
        ],
        axis=0
    )

    # 70/30 split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=SEED,
        stratify=y
    )

    # Balanced logistic regression, C=1
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED
        )
    )

    classifier.fit(
        X_train,
        y_train
    )

    predictions = classifier.predict(
        X_test
    )

    score = accuracy_score(
        y_test,
        predictions
    )

    return score


# Per-class target analysis
def per_class_analysis(
    method_results
):

    print("PER-CLASS TARGET ANALYSIS")

    for result in method_results:

        labels = result["target_labels"]
        predictions = result["target_predictions"]

        cm = confusion_matrix(
            labels,
            predictions,
            labels=np.arange(NUM_CLASSES)
        )

        per_class_accuracy = (
            cm.diagonal() /
            cm.sum(axis=1)
        )

        result["per_class_accuracy"] = (
            per_class_accuracy
        )

        print(
            f"\n{result['method']}"
        )

        for i, class_name in enumerate(CLASSES):

            print(
                f"  {class_name:10s}: "
                f"{per_class_accuracy[i]:.4f}"
            )

        result["confusion_matrix"] = cm


# Compare against Source-only
def compare_classes(method_results):

    source_only = method_results[0]

    source_acc = (
        source_only["per_class_accuracy"]
    )

    print("CLASS-SPECIFIC CHANGE RELATIVE TO SOURCE-ONLY")

    comparison = {}

    for result in method_results:

        if result["method"] == "Source-only":
            continue

        changes = (
            result["per_class_accuracy"]
            - source_acc
        )

        comparison[result["method"]] = changes

        best_idx = np.argmax(changes)
        worst_idx = np.argmin(changes)

        print(
            f"\n{result['method']}"
        )

        print(
            f"  Largest improvement: "
            f"{CLASSES[best_idx]} "
            f"({changes[best_idx]:+.4f})"
        )

        print(
            f"  Largest degradation: "
            f"{CLASSES[worst_idx]} "
            f"({changes[worst_idx]:+.4f})"
        )

        # Dominant confusion for improved/degraded classes
        cm = result["confusion_matrix"]

        # Largest improvement class
        row = cm[best_idx].copy()
        row[best_idx] = 0

        if row.sum() > 0:
            confusion_idx = np.argmax(row)

            print(
                f"  Dominant confusion for "
                f"{CLASSES[best_idx]}: "
                f"{CLASSES[confusion_idx]}"
            )

        # Largest degradation class
        row = cm[worst_idx].copy()
        row[worst_idx] = 0

        if row.sum() > 0:
            confusion_idx = np.argmax(row)

            print(
                f"  Dominant confusion for "
                f"{CLASSES[worst_idx]}: "
                f"{CLASSES[confusion_idx]}"
            )

    return comparison


# Main
def main():

    set_seed(SEED)

    print(f"Device: {DEVICE}")

    split_data = load_split_file()

    datasets = build_datasets(
        split_data
    )

    loaders = build_loaders(
        datasets
    )

    # Fixed checkpoints
    checkpoints = {
        "Source-only":
            RESULTS_DIR / "source_only_erm_best.pth",

        "DAN":
            RESULTS_DIR / "dan_best.pth",

        "DANN":
            RESULTS_DIR / "dann_best.pth",

        "CDAN":
            RESULTS_DIR / "cdan_best.pth",
    }

    # Final classification evaluation
    all_results = []

    for method, checkpoint in checkpoints.items():

        if not checkpoint.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}"
            )

        result = evaluate_method(
            method,
            checkpoint,
            loaders
        )

        all_results.append(
            result
        )

    # Target accuracy change relative to Source-only
    source_target_accuracy = (
        all_results[0]["target_accuracy"]
    )

    print("FINAL COMPARISON")

    print(
        f"{'Method':15s} "
        f"{'Src Acc':>10s} "
        f"{'Src F1':>10s} "
        f"{'Target Acc':>12s} "
        f"{'Target F1':>12s} "
        f"{'Target ΔAcc':>13s}"
    )

    for result in all_results:

        target_delta = (
            result["target_accuracy"]
            - source_target_accuracy
        )

        print(
            f"{result['method']:15s} "
            f"{result['mean_source_accuracy']:10.4f} "
            f"{result['mean_source_macro_f1']:10.4f} "
            f"{result['target_accuracy']:12.4f} "
            f"{result['target_macro_f1']:12.4f} "
            f"{target_delta:+13.4f}"
        )

        result["target_accuracy_change"] = (
            target_delta
        )

    # --------------------------------------------------------
    # Domain separability
    # --------------------------------------------------------

    print("DOMAIN SEPARABILITY")

    domain_scores = {}

    source_loaders = {
        domain: loaders[f"{domain}_val"]
        for domain in SOURCE_DOMAINS
    }

    for result in all_results:

        score = domain_separability(
            result["backbone"],
            source_loaders,
            loaders["target"]
        )

        domain_scores[
            result["method"]
        ] = score

        print(
            f"{result['method']:15s} | "
            f"Domain Separability: {score:.4f}"
        )

    # --------------------------------------------------------
    # Per-class analysis
    # --------------------------------------------------------

    per_class_analysis(
        all_results
    )

    compare_classes(
        all_results
    )

    # --------------------------------------------------------
    # Save numerical results
    # --------------------------------------------------------

    summary = []

    for result in all_results:

        summary.append(
            {
                "method": result["method"],
                "source_accuracy":
                    float(
                        result[
                            "mean_source_accuracy"
                        ]
                    ),
                "source_macro_f1":
                    float(
                        result[
                            "mean_source_macro_f1"
                        ]
                    ),
                "target_accuracy":
                    float(
                        result[
                            "target_accuracy"
                        ]
                    ),
                "target_macro_f1":
                    float(
                        result[
                            "target_macro_f1"
                        ]
                    ),
                "target_accuracy_change":
                    float(
                        result[
                            "target_accuracy_change"
                        ]
                    ),
                "domain_separability":
                    float(
                        domain_scores[
                            result["method"]
                        ]
                    ),
                "per_class_accuracy":
                    {
                        CLASSES[i]:
                            float(
                                result[
                                    "per_class_accuracy"
                                ][i]
                            )
                        for i in range(NUM_CLASSES)
                    }
            }
        )

    output_file = (
        RESULTS_DIR /
        "part5_final_results.json"
    )

    with open(output_file, "w") as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print(
        f"\nResults saved to: {output_file}"
    )


if __name__ == "__main__":
    main()