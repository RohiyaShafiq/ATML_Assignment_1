import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


SOURCE_DOMAINS = [
    "photo",
    "art_painting",
    "cartoon"
]

DOMAIN_TO_ID = {
    "photo": 0,
    "art_painting": 1,
    "cartoon": 2
}


@torch.no_grad()
def collect_source_features(
    backbone,
    source_val_loaders,
    device,
    samples_per_domain=None
):
    """
    Collect balanced backbone features from the
    three source validation domains.

    The backbone is frozen/evaluation-only.
    """

    backbone.eval()

    features = []
    domain_labels = []

    for domain in SOURCE_DOMAINS:

        collected = 0

        for batch in source_val_loaders[domain]:

            images, _, _ = batch

            images = images.to(device)

            batch_features = backbone(images)

            batch_features = batch_features.cpu().numpy()

            if samples_per_domain is not None:

                remaining = (
                    samples_per_domain - collected
                )

                if remaining <= 0:
                    break

                batch_features = batch_features[
                    :remaining
                ]

            features.append(batch_features)

            domain_labels.extend(
                [DOMAIN_TO_ID[domain]]
                * len(batch_features)
            )

            collected += len(batch_features)

            if (
                samples_per_domain is not None
                and collected >= samples_per_domain
            ):
                break

    features = np.concatenate(
        features,
        axis=0
    )

    domain_labels = np.asarray(
        domain_labels
    )

    return features, domain_labels


def source_domain_separability(
    backbone,
    source_val_loaders,
    device,
    seed=6304
):
    """
    70/30 stratified split followed by
    multinomial logistic regression with C=1.
    """

    # --------------------------------------------------------
    # Make the number of examples equal across domains
    # --------------------------------------------------------

    domain_sizes = []

    for domain in SOURCE_DOMAINS:

        domain_sizes.append(
            len(source_val_loaders[domain].dataset)
        )

    samples_per_domain = min(
        domain_sizes
    )

    X, y = collect_source_features(
        backbone,
        source_val_loaders,
        device,
        samples_per_domain=samples_per_domain
    )

    # --------------------------------------------------------
    # 70/30 split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y
    )

    # --------------------------------------------------------
    # Multinomial logistic regression
    # --------------------------------------------------------

    classifier = LogisticRegression(
        C=1.0,
        max_iter=2000,
        multi_class="multinomial",
        random_state=seed
    )

    classifier.fit(
        X_train,
        y_train
    )

    predictions = classifier.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return {
        "accuracy": float(accuracy),
        "chance": 1.0 / 3.0,
        "num_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }