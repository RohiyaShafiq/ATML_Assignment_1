import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


SOURCE_DOMAINS = [
    "photo",
    "art_painting",
    "cartoon",
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

    for images, labels, _ in loader:

        images = images.to(device)

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

    return accuracy, macro_f1


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

        accuracy, macro_f1 = evaluate_domain(
            backbone,
            classifier,
            source_val_loaders[domain],
            device
        )

        results[domain] = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1)
        }

        accuracies.append(accuracy)
        macro_f1s.append(macro_f1)

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