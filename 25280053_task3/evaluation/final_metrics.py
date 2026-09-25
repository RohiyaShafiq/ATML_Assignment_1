import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix
)


SOURCE_DOMAINS = [
    "photo",
    "art_painting",
    "cartoon"
]


@torch.no_grad()
def evaluate_domain(
    backbone,
    classifier,
    loader,
    device
):

    backbone.eval()
    classifier.eval()

    all_predictions = []
    all_labels = []

    for batch in loader:

        images = batch[0]
        labels = batch[1]

        images = images.to(device)

        features = backbone(images)

        logits = classifier(
            features
        )

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

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(7))
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm
    }


@torch.no_grad()
def evaluate_source_domains(
    backbone,
    classifier,
    source_val_loaders,
    device
):

    results = {}

    accuracies = []
    macro_f1s = []

    for domain in SOURCE_DOMAINS:

        metrics = evaluate_domain(
            backbone,
            classifier,
            source_val_loaders[domain],
            device
        )

        results[domain] = metrics

        accuracies.append(
            metrics["accuracy"]
        )

        macro_f1s.append(
            metrics["macro_f1"]
        )

    results["mean"] = {
        "accuracy": float(
            np.mean(accuracies)
        ),

        "macro_f1": float(
            np.mean(macro_f1s)
        )
    }

    results["worst_domain"] = {
        "accuracy": float(
            np.min(accuracies)
        ),

        "macro_f1": float(
            np.min(macro_f1s)
        )
    }

    return results