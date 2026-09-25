import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def domain_separability(
    source_features,
    target_features,
    seed=6304):

    X = np.concatenate(
        [source_features, target_features],
        axis=0)

    y = np.concatenate(
        [np.zeros(len(source_features)),
         np.ones(len(target_features))])

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=seed)

    classifier.fit(X, y)

    predictions = classifier.predict(X)
    accuracy = accuracy_score(y, predictions)

    return {
        "domain_accuracy": float(accuracy)}