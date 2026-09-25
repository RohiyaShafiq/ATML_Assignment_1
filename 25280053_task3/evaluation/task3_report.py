import os
import sys
import json
import random
import numpy as np
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

sys.path.insert(
    0,
    "/content/drive/MyDrive"
)

SPLIT_FILE = (
    "/content/drive/MyDrive/shared/splits/"
    "pacs_sketch_seed6304.json"
)

TASK2_RESULTS = (
    "/content/drive/MyDrive/results"
)

TASK3_RESULTS = (
    "/content/drive/MyDrive/task3/results"
)

SOURCE_DOMAINS = [
    "photo",
    "art_painting",
    "cartoon"
]

CLASS_NAMES = [
    "dog",
    "elephant",
    "giraffe",
    "guitar",
    "horse",
    "house",
    "person"
]

NUM_CLASSES = 7
BATCH_SIZE = 64
SEED = 6304

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMPORT YOUR PROJECT FILES
# ============================================================

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


# ============================================================
# SEED
# ============================================================

def set_seed(seed=6304):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    backbone = ResNet18Backbone().to(DEVICE)

    classifier = ClassifierHead(
        backbone.feature_dim,
        NUM_CLASSES
    ).to(DEVICE)

    return backbone, classifier


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_model(checkpoint_path):

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
# BUILD LOADERS
# ============================================================

def build_loaders():

    with open(SPLIT_FILE, "r") as f:
        split_data = json.load(f)

    source_loaders = {}

    for domain in SOURCE_DOMAINS:

        samples = split_data[
            "sources"
        ][domain]["val"]

        dataset = PACSDataset(
            samples,
            transform=get_transforms(
                train=False
            )
        )

        source_loaders[domain] = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

    # IMPORTANT:
    # Your JSON stores Sketch under target -> adaptation
    sketch_samples = split_data[
        "target"
    ]["adaptation"]

    sketch_dataset = PACSDataset(
        sketch_samples,
        transform=get_transforms(
            train=False
        )
    )

    sketch_loader = DataLoader(
        sketch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return source_loaders, sketch_loader


# ============================================================
# EVALUATE ONE DOMAIN
# ============================================================

@torch.no_grad()
def evaluate_domain(
    backbone,
    classifier,
    loader
):

    backbone.eval()
    classifier.eval()

    all_labels = []
    all_predictions = []

    for batch in loader:

        images = batch[0]
        labels = batch[1]

        images = images.to(DEVICE)

        features = backbone(images)
        logits = classifier(features)

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
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

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES))
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm
    }


# ============================================================
# SOURCE RESULTS
# ============================================================

def evaluate_sources(
    backbone,
    classifier,
    source_loaders
):

    results = {}

    accuracies = []
    f1s = []

    for domain in SOURCE_DOMAINS:

        result = evaluate_domain(
            backbone,
            classifier,
            source_loaders[domain]
        )

        results[domain] = result

        accuracies.append(
            result["accuracy"]
        )

        f1s.append(
            result["macro_f1"]
        )

    results["mean"] = {
        "accuracy": float(
            np.mean(accuracies)
        ),
        "macro_f1": float(
            np.mean(f1s)
        )
    }

    results["worst"] = {
        "accuracy": float(
            np.min(accuracies)
        ),
        "macro_f1": float(
            np.min(f1s)
        )
    }

    return results


# ============================================================
# COLLECT FEATURES FOR DOMAIN SEPARABILITY
# ============================================================

@torch.no_grad()
def collect_domain_features(
    backbone,
    source_loaders
):

    backbone.eval()

    features = []
    domain_labels = []

    # Balanced number of samples
    min_samples = min(
        len(source_loaders[d].dataset)
        for d in SOURCE_DOMAINS
    )

    for domain_id, domain in enumerate(
        SOURCE_DOMAINS
    ):

        collected = 0

        for batch in source_loaders[domain]:

            images = batch[0]

            images = images.to(DEVICE)

            batch_features = backbone(
                images
            )

            remaining = (
                min_samples - collected
            )

            batch_features = (
                batch_features[:remaining]
            )

            features.append(
                batch_features.cpu().numpy()
            )

            domain_labels.extend(
                [domain_id] *
                len(batch_features)
            )

            collected += len(
                batch_features
            )

            if collected >= min_samples:
                break

    features = np.concatenate(
        features,
        axis=0
    )

    domain_labels = np.array(
        domain_labels
    )

    return features, domain_labels


# ============================================================
# SOURCE-DOMAIN SEPARABILITY
# ============================================================

def compute_domain_separability(
    backbone,
    source_loaders
):

    X, y = collect_domain_features(
        backbone,
        source_loaders
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            stratify=y,
            random_state=SEED
        )
    )

    classifier = LogisticRegression(
        C=1.0,
        max_iter=2000,
        multi_class="multinomial",
        random_state=SEED
    )

    classifier.fit(
        X_train,
        y_train
    )

    accuracy = classifier.score(
        X_test,
        y_test
    )

    return float(accuracy)


# ============================================================
# FIXED SHARPNESS BATCH
# ============================================================

def build_fixed_sharpness_batch(
    source_loaders
):

    generator = torch.Generator()
    generator.manual_seed(SEED)

    images_list = []
    labels_list = []

    for domain in SOURCE_DOMAINS:

        dataset = source_loaders[
            domain
        ].dataset

        indices = torch.randperm(
            len(dataset),
            generator=generator
        )[:32].tolist()

        for index in indices:

            item = dataset[index]

            images_list.append(
                item[0]
            )

            labels_list.append(
                item[1]
            )

    images = torch.stack(
        images_list
    ).to(DEVICE)

    labels = torch.tensor(
        labels_list,
        dtype=torch.long
    ).to(DEVICE)

    return images, labels


# ============================================================
# SHARPNESS
# ============================================================

def compute_sharpness(
    backbone,
    classifier,
    fixed_batch,
    rho=0.05
):

    images, labels = fixed_batch

    backbone.eval()
    classifier.eval()

    parameters = list(
        backbone.parameters()
    ) + list(
        classifier.parameters()
    )

    # Original loss
    for parameter in parameters:
        parameter.grad = None

    features = backbone(images)
    logits = classifier(features)

    original_loss = torch.nn.functional.cross_entropy(
        logits,
        labels
    )

    original_loss.backward()

    gradients = [
        p.grad
        for p in parameters
        if p.grad is not None
    ]

    grad_norm = torch.sqrt(
        sum(
            torch.sum(g ** 2)
            for g in gradients
        )
    )

    scale = rho / (
        grad_norm + 1e-12
    )

    perturbations = []

    with torch.no_grad():

        for parameter in parameters:

            if parameter.grad is None:

                perturbations.append(
                    None
                )

                continue

            epsilon = (
                parameter.grad * scale
            )

            parameter.add_(
                epsilon
            )

            perturbations.append(
                epsilon
            )

    # Perturbed loss
    with torch.no_grad():

        features = backbone(images)
        logits = classifier(features)

        perturbed_loss = (
            torch.nn.functional.cross_entropy(
                logits,
                labels
            )
        )

    # Restore parameters
    with torch.no_grad():

        for parameter, epsilon in zip(
            parameters,
            perturbations
        ):

            if epsilon is not None:

                parameter.sub_(
                    epsilon
                )

    delta = (
        perturbed_loss.item()
        - original_loss.item()
    )

    return {
        "original_loss":
            float(original_loss.item()),

        "perturbed_loss":
            float(perturbed_loss.item()),

        "delta_sharp":
            float(delta)
    }


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

def class_accuracy(cm):

    correct = np.diag(cm)

    total = cm.sum(axis=1)

    return np.divide(
        correct,
        total,
        out=np.zeros(
            len(correct),
            dtype=float
        ),
        where=total != 0
    )


# ============================================================
# EVALUATE MAIN MODELS
# ============================================================

def evaluate_main_models():

    source_loaders, sketch_loader = (
        build_loaders()
    )

    fixed_batch = (
        build_fixed_sharpness_batch(
            source_loaders
        )
    )

    checkpoints = {

        "ERM":
            os.path.join(
                TASK2_RESULTS,
                "source_only_erm_best.pth"
            ),

        "DAN-DG":
            os.path.join(
                TASK3_RESULTS,
                "dan_dg_best.pth"
            ),

        "SAM":
            os.path.join(
                TASK3_RESULTS,
                "sam_best.pth"
            )
    }

    results = {}

    for name, checkpoint in checkpoints.items():

        print(
            f"\nEvaluating {name}..."
        )

        backbone, classifier = (
            load_model(checkpoint)
        )

        source_results = (
            evaluate_sources(
                backbone,
                classifier,
                source_loaders
            )
        )

        sketch_results = (
            evaluate_domain(
                backbone,
                classifier,
                sketch_loader
            )
        )

        separability = (
            compute_domain_separability(
                backbone,
                source_loaders
            )
        )

        sharpness = (
            compute_sharpness(
                backbone,
                classifier,
                fixed_batch,
                rho=0.05
            )
        )

        results[name] = {
            "source":
                source_results,

            "sketch":
                sketch_results,

            "separability":
                separability,

            "sharpness":
                sharpness
        }

    return results


# ============================================================
# MAIN COMPARISON TABLE
# ============================================================

def print_main_table(results):

    print("\n")
    print("=" * 150)
    print("TABLE 1 - ERM vs DAN-DG vs SAM")

    print(
        f"{'Model':<10}"
        f"{'Photo Acc':>11}"
        f"{'Photo F1':>11}"
        f"{'Art Acc':>11}"
        f"{'Art F1':>11}"
        f"{'Cartoon Acc':>13}"
        f"{'Cartoon F1':>13}"
        f"{'Mean Acc':>11}"
        f"{'Mean F1':>11}"
        f"{'Worst Acc':>12}"
        f"{'Worst F1':>12}"
        f"{'Sketch Acc':>12}"
        f"{'Sketch F1':>12}"
    )

    print("-" * 150)

    for name in [
        "ERM",
        "DAN-DG",
        "SAM"
    ]:

        s = results[
            name
        ]["source"]

        sk = results[
            name
        ]["sketch"]

        print(
            f"{name:<10}"
            f"{s['photo']['accuracy']:>11.4f}"
            f"{s['photo']['macro_f1']:>11.4f}"
            f"{s['art_painting']['accuracy']:>11.4f}"
            f"{s['art_painting']['macro_f1']:>11.4f}"
            f"{s['cartoon']['accuracy']:>13.4f}"
            f"{s['cartoon']['macro_f1']:>13.4f}"
            f"{s['mean']['accuracy']:>11.4f}"
            f"{s['mean']['macro_f1']:>11.4f}"
            f"{s['worst']['accuracy']:>12.4f}"
            f"{s['worst']['macro_f1']:>12.4f}"
            f"{sk['accuracy']:>12.4f}"
            f"{sk['macro_f1']:>12.4f}"
        )


# ============================================================
# SKETCH CHANGE RELATIVE TO ERM
# ============================================================

def print_sketch_changes(results):

    erm = results[
        "ERM"
    ]["sketch"]["accuracy"]

    print("\n")
    print("=" * 60)
    print("SKETCH ACCURACY CHANGE RELATIVE TO ERM")


    for name in [
        "ERM",
        "DAN-DG",
        "SAM"
    ]:

        acc = results[
            name
        ]["sketch"]["accuracy"]

        change = acc - erm

        print(
            f"{name:<10}"
            f" Accuracy = {acc:.4f}"
            f" Change = {change:+.4f}"
        )


# ============================================================
# DOMAIN SEPARABILITY
# ============================================================

def print_separability(results):

    print("\n")
    print("=" * 60)
    print("SOURCE-DOMAIN SEPARABILITY")

    print(
        f"{'Model':<12}"
        f"{'Separability':>18}"
        f"{'Chance':>15}"
    )

    print("-" * 50)

    for name in [
        "ERM",
        "DAN-DG",
        "SAM"
    ]:

        score = results[
            name
        ]["separability"]

        print(
            f"{name:<12}"
            f"{score:>18.4f}"
            f"{1/3:>15.4f}"
        )


# ============================================================
# SHARPNESS
# ============================================================

def print_sharpness(results):

    print("\n")
    print("=" * 65)
    print("COMMON LOCAL SHARPNESS PROXY")

    print(
        f"{'Model':<12}"
        f"{'Original':>18}"
        f"{'Perturbed':>18}"
        f"{'Delta Sharp':>18}"
    )

    print("-" * 65)

    for name in [
        "ERM",
        "DAN-DG",
        "SAM"
    ]:

        r = results[
            name
        ]["sharpness"]

        print(
            f"{name:<12}"
            f"{r['original_loss']:>18.6f}"
            f"{r['perturbed_loss']:>18.6f}"
            f"{r['delta_sharp']:>18.6f}"
        )


# ============================================================
# PER-CLASS SKETCH
# ============================================================

def print_per_class(results):

    erm = class_accuracy(
        results[
            "ERM"
        ]["sketch"]["confusion_matrix"]
    )

    dan = class_accuracy(
        results[
            "DAN-DG"
        ]["sketch"]["confusion_matrix"]
    )

    sam = class_accuracy(
        results[
            "SAM"
        ]["sketch"]["confusion_matrix"]
    )

    print("\n")
    print("=" * 85)
    print("PER-CLASS SKETCH ACCURACY")

    print(
        f"{'Class':<15}"
        f"{'ERM':>12}"
        f"{'DAN-DG':>12}"
        f"{'SAM':>12}"
        f"{'DAN-ERM':>14}"
        f"{'SAM-ERM':>14}"
    )

    print("-" * 85)

    for i, name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{name:<15}"
            f"{erm[i]:>12.4f}"
            f"{dan[i]:>12.4f}"
            f"{sam[i]:>12.4f}"
            f"{dan[i] - erm[i]:>14.4f}"
            f"{sam[i] - erm[i]:>14.4f}"
        )


# ============================================================
# CONFUSION MATRICES
# ============================================================

def print_confusion_matrices(results):

    for name in [
        "ERM",
        "DAN-DG",
        "SAM"
    ]:

        print("\n")
        print("=" * 60)
        print(
            f"{name} - SKETCH CONFUSION MATRIX"
        )

        print(
            results[
                name
            ]["sketch"]["confusion_matrix"]
        )


# ============================================================
# PART 5 - SAM SENSITIVITY
# ============================================================

def evaluate_sam_sensitivity():

    source_loaders, sketch_loader = (
        build_loaders()
    )

    fixed_batch = (
        build_fixed_sharpness_batch(
            source_loaders
        )
    )

    print("\n")
    print("=" * 75)
    print("TABLE 2 - SAM SENSITIVITY STUDY")

    print(
        f"{'rho':<10}"
        f"{'Source Mean F1':>18}"
        f"{'Source Worst F1':>20}"
        f"{'Sharpness':>18}"
        f"{'Sketch Acc':>15}"
        f"{'Sketch F1':>15}"
    )

    print("-" * 100)

    sensitivity_results = {}

    for rho in [
        0.01,
        0.05,
        0.1
    ]:

        checkpoint = os.path.join(
            TASK3_RESULTS,
            f"sam_rho_{rho}.pth"
        )

        if not os.path.exists(
            checkpoint
        ):

            print(
                f"Checkpoint missing: {checkpoint}"
            )

            continue

        backbone, classifier = (
            load_model(checkpoint)
        )

        source = evaluate_sources(
            backbone,
            classifier,
            source_loaders
        )

        sketch = evaluate_domain(
            backbone,
            classifier,
            sketch_loader
        )

        sharp = compute_sharpness(
            backbone,
            classifier,
            fixed_batch,
            rho=0.05
        )

        sensitivity_results[
            str(rho)
        ] = {
            "source_mean_f1":
                source["mean"]["macro_f1"],

            "source_worst_f1":
                source["worst"]["macro_f1"],

            "sharpness":
                sharp["delta_sharp"],

            "sketch_accuracy":
                sketch["accuracy"],

            "sketch_macro_f1":
                sketch["macro_f1"]
        }

        print(
            f"{rho:<10.2f}"
            f"{source['mean']['macro_f1']:>18.4f}"
            f"{source['worst']['macro_f1']:>20.4f}"
            f"{sharp['delta_sharp']:>18.6f}"
            f"{sketch['accuracy']:>15.4f}"
            f"{sketch['macro_f1']:>15.4f}"
        )

    return sensitivity_results


# ============================================================
# PART 5 PLOT
# ============================================================

def plot_sensitivity(
    sensitivity_results
):

    rhos = []
    sketch_acc = []
    sharpness = []

    for rho in [
        "0.01",
        "0.05",
        "0.1"
    ]:

        if rho not in sensitivity_results:
            continue

        rhos.append(
            float(rho)
        )

        sketch_acc.append(
            sensitivity_results[
                rho
            ]["sketch_accuracy"]
        )

        sharpness.append(
            sensitivity_results[
                rho
            ]["sharpness"]
        )

    plt.figure()

    plt.plot(
        rhos,
        sketch_acc,
        marker="o"
    )

    plt.xlabel("SAM rho")
    plt.ylabel("Sketch Accuracy")
    plt.title(
        "SAM Sensitivity: Sketch Accuracy"
    )

    plt.grid(True)

    plt.savefig(
        os.path.join(
            TASK3_RESULTS,
            "sam_sensitivity_sketch.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.figure()

    plt.plot(
        rhos,
        sharpness,
        marker="o"
    )

    plt.xlabel("SAM rho")
    plt.ylabel("Delta Sharpness")
    plt.title(
        "SAM Sensitivity: Local Sharpness"
    )

    plt.grid(True)

    plt.savefig(
        os.path.join(
            TASK3_RESULTS,
            "sam_sensitivity_sharpness.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    sensitivity_results
):

    output = {}

    for name, result in results.items():

        output[name] = {

            "source": {},

            "sketch": {
                "accuracy":
                    result["sketch"]["accuracy"],

                "macro_f1":
                    result["sketch"]["macro_f1"],

                "confusion_matrix":
                    result["sketch"]
                    ["confusion_matrix"]
                    .tolist()
            },

            "separability":
                result["separability"],

            "sharpness":
                result["sharpness"]
        }

        for domain in SOURCE_DOMAINS:

            output[name][
                "source"
            ][domain] = {

                "accuracy":
                    result["source"]
                    [domain]["accuracy"],

                "macro_f1":
                    result["source"]
                    [domain]["macro_f1"]
            }

        output[name][
            "source"
        ]["mean"] = result[
            "source"
        ]["mean"]

        output[name][
            "source"
        ]["worst"] = result[
            "source"
        ]["worst"]

    output[
        "SAM_sensitivity"
    ] = sensitivity_results

    path = os.path.join(
        TASK3_RESULTS,
        "task3_report_results.json"
    )

    with open(path, "w") as f:

        json.dump(
            output,
            f,
            indent=2
        )

    print(
        f"\nSaved results to:\n{path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    results = (
        evaluate_main_models()
    )

    print_main_table(
        results
    )

    print_sketch_changes(
        results
    )

    print_separability(
        results
    )

    print_sharpness(
        results
    )

    print_per_class(
        results
    )

    print_confusion_matrices(
        results
    )

    sensitivity_results = (
        evaluate_sam_sensitivity()
    )

    plot_sensitivity(
        sensitivity_results
    )

    save_results(
        results,
        sensitivity_results
    )


if __name__ == "__main__":
    main()