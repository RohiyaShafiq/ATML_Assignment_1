import numpy as np


def energy_unknownness(logits):

    logits = np.asarray(logits)

    max_logits = np.max(
        logits,
        axis=1,
        keepdims=True
    )

    logsumexp = (
        max_logits
        +
        np.log(
            np.sum(
                np.exp(
                    logits - max_logits
                ),
                axis=1,
                keepdims=True
            )
        )
    )

    return -logsumexp.squeeze(1)