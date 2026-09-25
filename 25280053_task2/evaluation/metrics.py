import numpy as np
from sklearn.metrics import (accuracy_score, f1_score)

def classification_metrics(y_true, y_pred):

    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0)}


def mean_source_macro_f1(source_metrics):

    values = [
        metrics["macro_f1"]
        for metrics in source_metrics.values()]

    return float(np.mean(values))