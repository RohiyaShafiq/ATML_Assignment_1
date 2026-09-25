# ============================================================
# task4/data/make_splits.py
# ============================================================

import numpy as np
from sklearn.model_selection import train_test_split


SEED = 6304


def make_stratified_split(targets, seed=SEED):

    targets = np.asarray(targets)
    indices = np.arange(len(targets))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.10,
        random_state=seed,
        stratify=targets
    )

    return train_indices, val_indices