import numpy as np

EPS = 1e-6


def fit_mahalanobis(
    features,
    labels,
    num_classes=10
):

    features = np.asarray(features)
    labels = np.asarray(labels)

    class_means = []

    # --------------------------------------------------------
    # Class means
    # --------------------------------------------------------

    for c in range(num_classes):

        class_features = features[
            labels == c
        ]

        if len(class_features) == 0:
            raise ValueError(
                f"No training features for class {c}"
            )

        mean_c = np.mean(
            class_features,
            axis=0
        )

        class_means.append(mean_c)

    class_means = np.stack(
        class_means,
        axis=0
    )

    # --------------------------------------------------------
    # Shared diagonal covariance
    #
    # Compute variance around the corresponding class mean.
    # --------------------------------------------------------

    centered = np.zeros_like(features)

    for c in range(num_classes):

        mask = labels == c

        centered[mask] = (
            features[mask]
            - class_means[c]
        )

    covariance_diag = np.mean(
        centered ** 2,
        axis=0
    )

    # Required numerical regularization
    covariance_diag += EPS

    return class_means, covariance_diag


def mahalanobis_unknownness(
    features,
    class_means,
    covariance_diag
):

    features = np.asarray(features)

    class_means = np.asarray(
        class_means
    )

    covariance_diag = np.asarray(
        covariance_diag
    )

    inverse_variance = (
        1.0 / covariance_diag
    )

    distances = []

    for mean in class_means:

        diff = (
            features - mean
        )

        distance = np.sum(
            diff ** 2
            * inverse_variance,
            axis=1
        )

        distances.append(distance)

    distances = np.stack(
        distances,
        axis=1
    )

    return np.min(
        distances,
        axis=1
    )