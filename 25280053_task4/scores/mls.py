import numpy as np


def mls_unknownness(logits):

    logits = np.asarray(logits)

    return -np.max(
        logits,
        axis=1
    )