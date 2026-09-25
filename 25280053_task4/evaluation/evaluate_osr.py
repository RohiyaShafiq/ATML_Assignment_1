import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    roc_curve
)

sys.path.append("/content/drive/MyDrive/task4")

from models.resnet_cifar import CIFARResNet18
from data.cifar100_unknowns import CIFAR100UnknownDataset


# ============================================================
# Configuration
# ============================================================

SEED = 6304

DATA_ROOT = "/content/drive/MyDrive/task4/data"

VANILLA_CHECKPOINT = (
    "/content/drive/MyDrive/task4/results/vanilla_best.pt"
)

GCSC_CHECKPOINT = (
    "/content/drive/MyDrive/task4/cache/gcsc_best.pt"
)

PROSER_CHECKPOINT = (
    "/content/drive/MyDrive/task4/cache/proser_best.pt"
)

RESULTS_DIR = (
    "/content/drive/MyDrive/task4/results"
)

FIGURES_DIR = os.path.join(
    RESULTS_DIR,
    "figures"
)

BATCH_SIZE = 128
NUM_WORKERS = 2

NUM_CLASSES = 10
NUM_DUMMY_CLASSES = 5

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Fixed CIFAR-100 unknown classes
# ============================================================

NEAR_CLASSES = [
    "bus",
    "pickup_truck",
    "motorcycle",
    "tractor",
    "wolf",
    "fox",
    "leopard",
    "camel"
]

FAR_CLASSES = [
    "bottle",
    "bowl",
    "chair",
    "clock",
    "keyboard",
    "mushroom",
    "sunflower",
    "wardrobe"
]


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

print("Device:", DEVICE)


# ============================================================
# CIFAR-10 normalization
# ============================================================

CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616
)


# ============================================================
# Evaluation transform
# ============================================================

eval_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    )
])


# ============================================================
# Load CIFAR-10 train partition
# ============================================================

base_train_dataset = datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    download=True,
    transform=None
)

targets = np.array(
    base_train_dataset.targets
)

indices = np.arange(
    len(base_train_dataset)
)


# ============================================================
# Recreate official 90/10 stratified split
# ============================================================

train_indices, val_indices = train_test_split(

    indices,

    test_size=0.10,

    random_state=SEED,

    stratify=targets
)


# ============================================================
# CIFAR-10 validation
# ============================================================

cifar10_train_eval = datasets.CIFAR10(
    root=DATA_ROOT,
    train=True,
    download=False,
    transform=eval_transform
)

val_dataset = Subset(
    cifar10_train_eval,
    val_indices
)


# ============================================================
# CIFAR-10 complete test set
# ============================================================

cifar10_test = datasets.CIFAR10(
    root=DATA_ROOT,
    train=False,
    download=True,
    transform=eval_transform
)


# ============================================================
# CIFAR-100 near/far unknowns
# ============================================================

near_dataset = CIFAR100UnknownDataset(
    root=DATA_ROOT,
    class_names=NEAR_CLASSES,
    transform=eval_transform
)

far_dataset = CIFAR100UnknownDataset(
    root=DATA_ROOT,
    class_names=FAR_CLASSES,
    transform=eval_transform
)


# ============================================================
# Verify required sizes
# ============================================================

print()
print("Dataset sizes:")
print("CIFAR-10 validation:", len(val_dataset))
print("CIFAR-10 test      :", len(cifar10_test))
print("Near unknowns      :", len(near_dataset))
print("Far unknowns       :", len(far_dataset))

assert len(val_dataset) == 5000
assert len(cifar10_test) == 10000
assert len(near_dataset) == 800
assert len(far_dataset) == 800


# ============================================================
# DataLoaders
# ============================================================

def make_loader(dataset):

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )


val_loader = make_loader(val_dataset)
test_loader = make_loader(cifar10_test)
near_loader = make_loader(near_dataset)
far_loader = make_loader(far_dataset)


# ============================================================
# Load Vanilla
# ============================================================

def load_vanilla():

    model = CIFARResNet18(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    checkpoint = torch.load(
        VANILLA_CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Loaded Vanilla checkpoint:",
        VANILLA_CHECKPOINT
    )

    print(
        "Vanilla checkpoint validation accuracy:",
        checkpoint.get("val_accuracy", "N/A")
    )

    return model


# ============================================================
# Load GCSC
# ============================================================

def load_gcsc():

    model = CIFARResNet18(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    checkpoint = torch.load(
        GCSC_CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Loaded GCSC checkpoint:",
        GCSC_CHECKPOINT
    )

    print(
        "GCSC checkpoint validation accuracy:",
        checkpoint.get("val_accuracy", "N/A")
    )

    return model


# ============================================================
# Load PROSER
# ============================================================

def load_proser():

    model = CIFARResNet18(
        num_classes=NUM_CLASSES + NUM_DUMMY_CLASSES
    ).to(DEVICE)

    checkpoint = torch.load(
        PROSER_CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Loaded PROSER checkpoint:",
        PROSER_CHECKPOINT
    )

    print(
        "PROSER checkpoint validation accuracy:",
        checkpoint.get("val_accuracy", "N/A")
    )

    return model


# ============================================================
# Extract logits and features
# ============================================================

def extract_outputs(model, loader):

    model.eval()

    all_features = []
    all_logits = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            features, logits = model(
                images,
                return_features=True
            )

            all_features.append(
                features.cpu()
            )

            all_logits.append(
                logits.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

    return (
        torch.cat(all_features, dim=0),
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0)
    )


# ============================================================
# Extract outputs for a model
# ============================================================

def extract_all(model):

    print("  Extracting validation outputs...")
    val_features, val_logits, val_labels = (
        extract_outputs(
            model,
            val_loader
        )
    )

    print("  Extracting test outputs...")
    test_features, test_logits, test_labels = (
        extract_outputs(
            model,
            test_loader
        )
    )

    print("  Extracting near unknown outputs...")
    near_features, near_logits, near_labels = (
        extract_outputs(
            model,
            near_loader
        )
    )

    print("  Extracting far unknown outputs...")
    far_features, far_logits, far_labels = (
        extract_outputs(
            model,
            far_loader
        )
    )

    return {
        "val_features": val_features,
        "val_logits": val_logits,
        "val_labels": val_labels,

        "test_features": test_features,
        "test_logits": test_logits,
        "test_labels": test_labels,

        "near_features": near_features,
        "near_logits": near_logits,
        "near_labels": near_labels,

        "far_features": far_features,
        "far_logits": far_logits,
        "far_labels": far_labels
    }


# ============================================================
# Score functions
#
# Larger = more novel
# ============================================================

def score_msp(logits):

    known_logits = logits[
        :, :NUM_CLASSES
    ]

    probabilities = torch.softmax(
        known_logits,
        dim=1
    )

    return 1.0 - probabilities.max(
        dim=1
    ).values


def score_mls(logits):

    known_logits = logits[
        :, :NUM_CLASSES
    ]

    return -known_logits.max(
        dim=1
    ).values


def score_energy(logits):

    known_logits = logits[
        :, :NUM_CLASSES
    ]

    return -torch.logsumexp(
        known_logits,
        dim=1
    )


# ============================================================
# Mahalanobis statistics
#
# IMPORTANT:
# Means/covariance are computed ONLY from
# unaugmented CIFAR-10 training features.
# ============================================================

def compute_mahalanobis_statistics(
    vanilla_model,
    train_indices
):

    print()
    print(
        "Computing Mahalanobis statistics "
        "from CIFAR-10 training features..."
    )

    train_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=False,
        transform=eval_transform
    )

    train_subset = Subset(
        train_dataset,
        train_indices
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    features = []
    labels = []

    vanilla_model.eval()

    with torch.no_grad():

        for images, batch_labels in train_loader:

            images = images.to(DEVICE)

            batch_features, _ = vanilla_model(
                images,
                return_features=True
            )

            features.append(
                batch_features.cpu()
            )

            labels.append(
                batch_labels.cpu()
            )

    features = torch.cat(
        features,
        dim=0
    )

    labels = torch.cat(
        labels,
        dim=0
    )

    feature_dim = features.shape[1]

    means = []

    for class_id in range(NUM_CLASSES):

        class_features = features[
            labels == class_id
        ]

        class_mean = class_features.mean(
            dim=0
        )

        means.append(
            class_mean
        )

    means = torch.stack(
        means,
        dim=0
    )

    # Shared diagonal covariance
    centered = (
        features
        -
        means[labels]
    )

    variance = (
        centered ** 2
    ).mean(
        dim=0
    )

    # Required 1e-6 regularization
    variance = variance + 1e-6

    inverse_variance = (
        1.0 / variance
    )

    print(
        "Mahalanobis feature dimension:",
        feature_dim
    )

    return (
        means,
        inverse_variance
    )


def score_mahalanobis(
    features,
    means,
    inverse_variance
):

    # features:
    # [N, D]
    #
    # means:
    # [10, D]

    distances = []

    for class_id in range(NUM_CLASSES):

        diff = (
            features
            -
            means[class_id]
        )

        distance = (
            diff ** 2
            *
            inverse_variance
        ).sum(
            dim=1
        )

        distances.append(
            distance
        )

    distances = torch.stack(
        distances,
        dim=1
    )

    # Minimum distance to any known class
    return distances.min(
        dim=1
    ).values


# ============================================================
# Validation-calibrated threshold
#
# Requirement:
# threshold = 95th percentile of unknownness
# on CIFAR-10 validation.
#
# Accept if u(x) <= threshold.
# ============================================================

def validation_threshold(scores):

    return np.percentile(
        scores.numpy(),
        95
    )


# ============================================================
# Compute rejection statistics
# ============================================================

def rejection_statistics(
    val_scores,
    test_scores,
    near_scores,
    far_scores
):

    threshold = validation_threshold(
        val_scores
    )

    test_acceptance = (
        test_scores <= threshold
    ).float().mean().item()

    near_rejection = (
        near_scores > threshold
    ).float().mean().item()

    far_rejection = (
        far_scores > threshold
    ).float().mean().item()

    near_fpr = 1.0 - near_rejection

    far_fpr = 1.0 - far_rejection

    return {
        "threshold": threshold,
        "test_acceptance": test_acceptance,
        "near_rejection": near_rejection,
        "far_rejection": far_rejection,
        "near_fpr": near_fpr,
        "far_fpr": far_fpr
    }


# ============================================================
# AUROC
#
# Known = negative
# Unknown = positive
# ============================================================

def compute_auroc(
    known_scores,
    unknown_scores
):

    y_true = np.concatenate([
        np.zeros(
            len(known_scores)
        ),
        np.ones(
            len(unknown_scores)
        )
    ])

    y_score = np.concatenate([
        known_scores.numpy(),
        unknown_scores.numpy()
    ])

    return roc_auc_score(
        y_true,
        y_score
    )


# ============================================================
# Complete score evaluation
# ============================================================

def evaluate_score(
    score_name,
    val_scores,
    test_scores,
    near_scores,
    far_scores
):

    near_auroc = compute_auroc(
        test_scores,
        near_scores
    )

    far_auroc = compute_auroc(
        test_scores,
        far_scores
    )

    all_unknown_scores = torch.cat([
        near_scores,
        far_scores
    ])

    all_auroc = compute_auroc(
        test_scores,
        all_unknown_scores
    )

    rejection = rejection_statistics(
        val_scores,
        test_scores,
        near_scores,
        far_scores
    )

    result = {
        "score": score_name,

        "near_auroc": near_auroc,

        "far_auroc": far_auroc,

        "all_auroc": all_auroc,

        "threshold": rejection[
            "threshold"
        ],

        "test_acceptance": rejection[
            "test_acceptance"
        ],

        "near_rejection": rejection[
            "near_rejection"
        ],

        "far_rejection": rejection[
            "far_rejection"
        ],

        "near_fpr95": rejection[
            "near_fpr"
        ],

        "far_fpr95": rejection[
            "far_fpr"
        ]
    }

    return result


# ============================================================
# Print result
# ============================================================

def print_result(result):

    print(
        f"{result['score']:<25}"
        f"{result['near_auroc']:.4f}    "
        f"{result['far_auroc']:.4f}    "
        f"{result['all_auroc']:.4f}    "
        f"{result['threshold']:.6f}    "
        f"{result['test_acceptance']:.4f}    "
        f"{result['near_rejection']:.4f}    "
        f"{result['far_rejection']:.4f}"
    )


# ============================================================
# Evaluate Vanilla scores
# ============================================================

def evaluate_vanilla():

    print()
    print("============================================================")
    print("VANILLA")
    print("============================================================")

    model = load_vanilla()

    outputs = extract_all(
        model
    )

    # --------------------------------------------------------
    # Mahalanobis statistics from CIFAR-10 training only
    # --------------------------------------------------------

    means, inverse_variance = (
        compute_mahalanobis_statistics(
            model,
            train_indices
        )
    )

    results = []

    # --------------------------------------------------------
    # MSP
    # --------------------------------------------------------

    val_msp = score_msp(
        outputs["val_logits"]
    )

    test_msp = score_msp(
        outputs["test_logits"]
    )

    near_msp = score_msp(
        outputs["near_logits"]
    )

    far_msp = score_msp(
        outputs["far_logits"]
    )

    result = evaluate_score(
        "MSP",
        val_msp,
        test_msp,
        near_msp,
        far_msp
    )

    results.append(result)

    # --------------------------------------------------------
    # MLS
    # --------------------------------------------------------

    val_mls = score_mls(
        outputs["val_logits"]
    )

    test_mls = score_mls(
        outputs["test_logits"]
    )

    near_mls = score_mls(
        outputs["near_logits"]
    )

    far_mls = score_mls(
        outputs["far_logits"]
    )

    result = evaluate_score(
        "MLS",
        val_mls,
        test_mls,
        near_mls,
        far_mls
    )

    results.append(result)

    # --------------------------------------------------------
    # Energy
    # --------------------------------------------------------

    val_energy = score_energy(
        outputs["val_logits"]
    )

    test_energy = score_energy(
        outputs["test_logits"]
    )

    near_energy = score_energy(
        outputs["near_logits"]
    )

    far_energy = score_energy(
        outputs["far_logits"]
    )

    result = evaluate_score(
        "Energy",
        val_energy,
        test_energy,
        near_energy,
        far_energy
    )

    results.append(result)

    # --------------------------------------------------------
    # Mahalanobis
    # --------------------------------------------------------

    val_mahal = score_mahalanobis(
        outputs["val_features"],
        means,
        inverse_variance
    )

    test_mahal = score_mahalanobis(
        outputs["test_features"],
        means,
        inverse_variance
    )

    near_mahal = score_mahalanobis(
        outputs["near_features"],
        means,
        inverse_variance
    )

    far_mahal = score_mahalanobis(
        outputs["far_features"],
        means,
        inverse_variance
    )

    result = evaluate_score(
        "Mahalanobis",
        val_mahal,
        test_mahal,
        near_mahal,
        far_mahal
    )

    results.append(result)

    return (
        outputs,
        results,
        {
            "msp": (
                val_msp,
                test_msp,
                near_msp,
                far_msp
            ),
            "mls": (
                val_mls,
                test_mls,
                near_mls,
                far_mls
            ),
            "energy": (
                val_energy,
                test_energy,
                near_energy,
                far_energy
            ),
            "mahalanobis": (
                val_mahal,
                test_mahal,
                near_mahal,
                far_mahal
            )
        }
    )


# ============================================================
# Evaluate GCSC with MLS
# ============================================================

def evaluate_gcsc():

    print()
    print("============================================================")
    print("GCSC")
    print("============================================================")

    model = load_gcsc()

    outputs = extract_all(
        model
    )

    val_mls = score_mls(
        outputs["val_logits"]
    )

    test_mls = score_mls(
        outputs["test_logits"]
    )

    near_mls = score_mls(
        outputs["near_logits"]
    )

    far_mls = score_mls(
        outputs["far_logits"]
    )

    result = evaluate_score(
        "GCSC - MLS",
        val_mls,
        test_mls,
        near_mls,
        far_mls
    )

    return outputs, result


# ============================================================
# PROSER placeholder score
#
# Uses strongest dummy response.
#
# The validation threshold is calibrated ONLY using
# CIFAR-10 validation data.
# ============================================================

def score_proser_placeholder(logits):

    known_logits = logits[
        :, :NUM_CLASSES
    ]

    dummy_logits = logits[
        :, NUM_CLASSES:
    ]

    strongest_dummy = dummy_logits.max(
        dim=1
    ).values

    strongest_known = known_logits.max(
        dim=1
    ).values

    # Higher = more novel
    #
    # A dummy classifier should respond more strongly
    # than the known classifiers for an unknown sample.
    return (
        strongest_dummy
        -
        strongest_known
    )


# ============================================================
# Evaluate PROSER
# ============================================================

def evaluate_proser():

    print()
    print("============================================================")
    print("PROSER")
    print("============================================================")

    model = load_proser()

    outputs = extract_all(
        model
    )

    # --------------------------------------------------------
    # PROSER MLS
    # --------------------------------------------------------

    val_mls = score_mls(
        outputs["val_logits"]
    )

    test_mls = score_mls(
        outputs["test_logits"]
    )

    near_mls = score_mls(
        outputs["near_logits"]
    )

    far_mls = score_mls(
        outputs["far_logits"]
    )

    mls_result = evaluate_score(
        "PROSER - MLS",
        val_mls,
        test_mls,
        near_mls,
        far_mls
    )

    # --------------------------------------------------------
    # PROSER placeholder score
    # --------------------------------------------------------

    val_placeholder = score_proser_placeholder(
        outputs["val_logits"]
    )

    test_placeholder = score_proser_placeholder(
        outputs["test_logits"]
    )

    near_placeholder = score_proser_placeholder(
        outputs["near_logits"]
    )

    far_placeholder = score_proser_placeholder(
        outputs["far_logits"]
    )

    placeholder_result = evaluate_score(
        "PROSER - Placeholder",
        val_placeholder,
        test_placeholder,
        near_placeholder,
        far_placeholder
    )

    return (
        outputs,
        mls_result,
        placeholder_result
    )


# ============================================================
# PROSER CSA
#
# Classification uses ONLY ten known-class logits.
# Dummy classifiers are NOT used for classification accuracy.
# ============================================================

def compute_proser_csa(
    test_logits,
    test_labels
):

    known_logits = test_logits[
        :, :NUM_CLASSES
    ]

    predictions = known_logits.argmax(
        dim=1
    )

    correct = (
        predictions
        ==
        test_labels
    ).float().mean().item()

    return correct


# ============================================================
# Failure analysis
#
# Uses Vanilla MLS threshold.
# ============================================================

def failure_analysis(
    vanilla_outputs,
    vanilla_mls_scores,
    vanilla_mls_result
):

    threshold = vanilla_mls_result[
        "threshold"
    ]

    near_scores = vanilla_mls_scores[
        2
    ]

    far_scores = vanilla_mls_scores[
        3
    ]

    near_labels = vanilla_outputs[
        "near_labels"
    ]

    far_labels = vanilla_outputs[
        "far_labels"
    ]

    near_logits = vanilla_outputs[
        "near_logits"
    ]

    far_logits = vanilla_outputs[
        "far_logits"
    ]

    # --------------------------------------------------------
    # CIFAR-100 class names
    # --------------------------------------------------------

    cifar100_classes = datasets.CIFAR100(
        root=DATA_ROOT,
        train=False,
        download=True
    ).classes

    cifar10_classes = datasets.CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=True
    ).classes

    # --------------------------------------------------------
    # Incorrectly accepted = score <= threshold
    # --------------------------------------------------------

    near_accepted = torch.nonzero(
        near_scores <= threshold,
        as_tuple=False
    ).flatten()

    far_accepted = torch.nonzero(
        far_scores <= threshold,
        as_tuple=False
    ).flatten()

    # Sort by score so the most confidently accepted
    # unknowns are inspected first.
    near_sorted = near_accepted[
        torch.argsort(
            near_scores[near_accepted]
        )
    ]

    far_sorted = far_accepted[
        torch.argsort(
            far_scores[far_accepted]
        )
    ]

    print()
    print("============================================================")
    print("FAILURE ANALYSIS")
    print("============================================================")

    print()
    print(
        f"Vanilla MLS threshold: {threshold:.6f}"
    )

    # --------------------------------------------------------
    # Near failures
    # --------------------------------------------------------

    print()
    print("Three incorrectly accepted NEAR unknowns:")
    print()

    print(
        f"{'CIFAR-100 class':<20}"
        f"{'Predicted CIFAR-10':<22}"
        f"{'Score':<14}"
        f"{'Threshold':<14}"
    )

    print("-" * 70)

    for idx in near_sorted[:3]:

        idx = idx.item()

        cifar100_class = cifar100_classes[
            near_labels[idx].item()
        ]

        predicted_class = near_logits[
            idx,
            :NUM_CLASSES
        ].argmax().item()

        predicted_name = cifar10_classes[
            predicted_class
        ]

        score = near_scores[
            idx
        ].item()

        print(
            f"{cifar100_class:<20}"
            f"{predicted_name:<22}"
            f"{score:<14.6f}"
            f"{threshold:<14.6f}"
        )

    # --------------------------------------------------------
    # Far failures
    # --------------------------------------------------------

    print()
    print("Three incorrectly accepted FAR unknowns:")
    print()

    print(
        f"{'CIFAR-100 class':<20}"
        f"{'Predicted CIFAR-10':<22}"
        f"{'Score':<14}"
        f"{'Threshold':<14}"
    )

    print("-" * 70)

    for idx in far_sorted[:3]:

        idx = idx.item()

        cifar100_class = cifar100_classes[
            far_labels[idx].item()
        ]

        predicted_class = far_logits[
            idx,
            :NUM_CLASSES
        ].argmax().item()

        predicted_name = cifar10_classes[
            predicted_class
        ]

        score = far_scores[
            idx
        ].item()

        print(
            f"{cifar100_class:<20}"
            f"{predicted_name:<22}"
            f"{score:<14.6f}"
            f"{threshold:<14.6f}"
        )

    return {
        "threshold": threshold,
        "near_indices": near_sorted[:3],
        "far_indices": far_sorted[:3]
    }


# ============================================================
# ROC curves
#
# Required:
# MSP, MLS, Mahalanobis
# on one compact multi-panel figure.
# ============================================================

def plot_roc_curves(
    vanilla_scores
):

    os.makedirs(
        FIGURES_DIR,
        exist_ok=True
    )

    score_names = [
        "MSP",
        "MLS",
        "Mahalanobis"
    ]

    score_keys = [
        "msp",
        "mls",
        "mahalanobis"
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5)
    )

    for ax, name, key in zip(
        axes,
        score_names,
        score_keys
    ):

        (
            _,
            test_scores,
            near_scores,
            far_scores
        ) = vanilla_scores[key]

        # ----------------------------------------------------
        # Known vs Near
        # ----------------------------------------------------

        y_near = np.concatenate([
            np.zeros(
                len(test_scores)
            ),
            np.ones(
                len(near_scores)
            )
        ])

        s_near = np.concatenate([
            test_scores.numpy(),
            near_scores.numpy()
        ])

        fpr_near, tpr_near, _ = roc_curve(
            y_near,
            s_near
        )

        auc_near = roc_auc_score(
            y_near,
            s_near
        )

        # ----------------------------------------------------
        # Known vs Far
        # ----------------------------------------------------

        y_far = np.concatenate([
            np.zeros(
                len(test_scores)
            ),
            np.ones(
                len(far_scores)
            )
        ])

        s_far = np.concatenate([
            test_scores.numpy(),
            far_scores.numpy()
        ])

        fpr_far, tpr_far, _ = roc_curve(
            y_far,
            s_far
        )

        auc_far = roc_auc_score(
            y_far,
            s_far
        )

        ax.plot(
            fpr_near,
            tpr_near,
            label=f"Near AUROC = {auc_near:.3f}"
        )

        ax.plot(
            fpr_far,
            tpr_far,
            label=f"Far AUROC = {auc_far:.3f}"
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        ax.set_title(
            name
        )

        ax.set_xlabel(
            "False Positive Rate"
        )

        ax.set_ylabel(
            "True Positive Rate"
        )

        ax.legend(
            fontsize=8
        )

        ax.grid(
            alpha=0.3
        )

    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "part6_vanilla_roc_curves.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "Saved ROC figure:"
    )

    print(
        output_path
    )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Vanilla
    # ========================================================

    (
        vanilla_outputs,
        vanilla_results,
        vanilla_scores
    ) = evaluate_vanilla()


    # ========================================================
    # GCSC
    # ========================================================

    (
        gcsc_outputs,
        gcsc_result
    ) = evaluate_gcsc()


    # ========================================================
    # PROSER
    # ========================================================

    (
        proser_outputs,
        proser_mls_result,
        proser_placeholder_result
    ) = evaluate_proser()


    # ========================================================
    # Print Vanilla score comparison
    # ========================================================

    print()
    print("============================================================")
    print("VANILLA SCORE COMPARISON")

    print(
        f"{'Score':<25}"
        f"{'Near AUROC':<13}"
        f"{'Far AUROC':<13}"
        f"{'All AUROC':<13}"
        f"{'Threshold':<14}"
        f"{'Test Accept':<14}"
        f"{'Near Reject':<14}"
        f"{'Far Reject':<14}"
    )

    print("-" * 120)

    for result in vanilla_results:

        print_result(
            result
        )


    # ========================================================
    # Trained-model comparison
    # ========================================================

    print()
    print("============================================================")
    print("TRAINED-MODEL COMPARISON")

    print(
        f"{'Model / Score':<25}"
        f"{'Near AUROC':<13}"
        f"{'Far AUROC':<13}"
        f"{'All AUROC':<13}"
        f"{'Threshold':<14}"
        f"{'Test Accept':<14}"
        f"{'Near Reject':<14}"
        f"{'Far Reject':<14}"
    )

    print("-" * 120)

    print_result(
        {
            "score": "Vanilla - MLS",
            **next(
                r
                for r in vanilla_results
                if r["score"] == "MLS"
            )
        }
    )

    print_result(
        gcsc_result
    )

    print_result(
        proser_mls_result
    )

    print_result(
        proser_placeholder_result
    )


    # ========================================================
    # PROSER CSA
    # ========================================================

    proser_csa = compute_proser_csa(
        proser_outputs["test_logits"],
        proser_outputs["test_labels"]
    )

    print()
    print(
        "PROSER CSA "
        "(10 known-class logits only): "
        f"{proser_csa:.4f}"
    )


    # ========================================================
    # Plot ROC curves
    # ========================================================

    plot_roc_curves(
        vanilla_scores
    )


    # ========================================================
    # Failure analysis
    # ========================================================

    failure_info = failure_analysis(
        vanilla_outputs,
        vanilla_scores["mls"],
        next(
            r
            for r in vanilla_results
            if r["score"] == "MLS"
        )
    )


    # ========================================================
    # Save numerical results
    # ========================================================

    torch.save(
        {
            "vanilla_results": vanilla_results,

            "gcsc_result": gcsc_result,

            "proser_mls_result":
                proser_mls_result,

            "proser_placeholder_result":
                proser_placeholder_result,

            "proser_csa":
                proser_csa,

            "failure_analysis":
                failure_info
        },

        os.path.join(
            RESULTS_DIR,
            "part6_results.pt"
        )
    )


    print()
    print("============================================================")
    print("PART 6 COMPLETE")


    print()
    print(
        "Results saved to:"
    )

    print(
        os.path.join(
            RESULTS_DIR,
            "part6_results.pt"
        )
    )


if __name__ == "__main__":
    main()