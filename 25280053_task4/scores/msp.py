# ============================================================
# task4/scores/msp.py
# ============================================================

import numpy as np


def msp_unknownness(logits):
    """
    u_MSP(x) = 1 - max_k p_k(x)

    Larger value = more novel.
    """

    logits = np.asarray(logits)

    logits_shifted = (
        logits - np.max(
            logits,
            axis=1,
            keepdims=True
        )
    )

    probabilities = (
        np.exp(logits_shifted)
        /
        np.sum(
            np.exp(logits_shifted),
            axis=1,
            keepdims=True
        )
    )

    max_probability = np.max(
        probabilities,
        axis=1
    )

    return 1.0 - max_probability