import os
import numpy as np

from scores.msp import msp_unknownness
from scores.mls import mls_unknownness
from scores.energy import energy_unknownness

from scores.mahalanobis import (
    fit_mahalanobis,
    mahalanobis_unknownness
)


CACHE_DIR = (
    "/content/drive/MyDrive/task4/cache"
)

RESULTS_DIR = (
    "/content/drive/MyDrive/task4/results"
)


def load_outputs(name):

    features = np.load(
        os.path.join(
            CACHE_DIR,
            f"{name}_features.npy"
        )
    )

    logits = np.load(
        os.path.join(
            CACHE_DIR,
            f"{name}_logits.npy"
        )
    )

    labels = np.load(
        os.path.join(
            CACHE_DIR,
            f"{name}_labels.npy"
        )
    )

    return features, logits, labels


# ============================================================
# Save score
# ============================================================

def save_score(
    dataset_name,
    score_name,
    scores
):

    path = os.path.join(
        CACHE_DIR,
        f"{dataset_name}_{score_name}.npy"
    )

    np.save(
        path,
        scores
    )

    return path


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load fixed model outputs
    # --------------------------------------------------------

    (
        train_features,
        train_logits,
        train_labels
    ) = load_outputs(
        "cifar10_train"
    )

    (
        val_features,
        val_logits,
        val_labels
    ) = load_outputs(
        "cifar10_val"
    )

    (
        test_features,
        test_logits,
        test_labels
    ) = load_outputs(
        "cifar10_test"
    )

    (
        near_features,
        near_logits,
        near_labels
    ) = load_outputs(
        "cifar100_near"
    )

    (
        far_features,
        far_logits,
        far_labels
    ) = load_outputs(
        "cifar100_far"
    )

    # ========================================================
    # Part 1: CIFAR-10 test accuracy
    # ========================================================

    test_predictions = np.argmax(
        test_logits,
        axis=1
    )

    test_accuracy = np.mean(
        test_predictions == test_labels
    )

    print(
        "\nCIFAR-10 Test Accuracy:",
        f"{test_accuracy:.4f}"
    )

    # ========================================================
    # Part 2
    # ========================================================

    # --------------------------------------------------------
    # MSP
    # --------------------------------------------------------

    print("\nCalculating MSP...")

    datasets_logits = {
        "cifar10_train": train_logits,
        "cifar10_val": val_logits,
        "cifar10_test": test_logits,
        "cifar100_near": near_logits,
        "cifar100_far": far_logits,
    }

    for dataset_name, logits in datasets_logits.items():

        scores = msp_unknownness(
            logits
        )

        save_score(
            dataset_name,
            "msp",
            scores
        )

    # --------------------------------------------------------
    # MLS
    # --------------------------------------------------------

    print("Calculating MLS...")

    for dataset_name, logits in datasets_logits.items():

        scores = mls_unknownness(
            logits
        )

        save_score(
            dataset_name,
            "mls",
            scores
        )

    # --------------------------------------------------------
    # Energy
    # --------------------------------------------------------

    print("Calculating Energy...")

    for dataset_name, logits in datasets_logits.items():

        scores = energy_unknownness(
            logits
        )

        save_score(
            dataset_name,
            "energy",
            scores
        )

    # ========================================================
    # Mahalanobis
    # ========================================================

    print(
        "Fitting Mahalanobis statistics "
        "from CIFAR-10 training features..."
    )

    class_means, covariance_diag = (
        fit_mahalanobis(
            train_features,
            train_labels,
            num_classes=10
        )
    )

    # Save statistics
    np.save(
        os.path.join(
            CACHE_DIR,
            "mahalanobis_class_means.npy"
        ),
        class_means
    )

    np.save(
        os.path.join(
            CACHE_DIR,
            "mahalanobis_covariance_diag.npy"
        ),
        covariance_diag
    )

    datasets_features = {
        "cifar10_train": train_features,
        "cifar10_val": val_features,
        "cifar10_test": test_features,
        "cifar100_near": near_features,
        "cifar100_far": far_features,
    }

    for dataset_name, features in datasets_features.items():

        scores = mahalanobis_unknownness(
            features,
            class_means,
            covariance_diag
        )

        save_score(
            dataset_name,
            "mahalanobis",
            scores
        )

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 60)
    print("PART 1 + PART 2 COMPLETE")

    print(
        f"CIFAR-10 test accuracy: "
        f"{test_accuracy:.4f}"
    )

    print("\nSaved novelty scores:")

    for score_name in [
        "msp",
        "mls",
        "energy",
        "mahalanobis"
    ]:

        print(
            f"  {score_name}"
        )

    print(
        "\nAll scores use the same frozen "
        "Vanilla checkpoint outputs."
    )


if __name__ == "__main__":
    main()