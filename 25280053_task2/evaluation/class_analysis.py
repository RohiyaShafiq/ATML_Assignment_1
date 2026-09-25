import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report)


def get_confusion_matrix(y_true, y_pred, num_classes=7):

    return confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(num_classes))


def get_classification_report(
    y_true,
    y_pred,
    class_names):

    return classification_report(
        y_true,
        y_pred,
        labels=np.arange(
            len(class_names)),
        target_names=class_names,
        zero_division=0)