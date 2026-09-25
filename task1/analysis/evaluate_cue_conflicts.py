from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))


from configs.config import SEED

from models.backbones import (
    ResNet50Backbone,
    ViTB16Backbone,
    CLIPViTB32Backbone,
    LinearClassifier,
)

from analysis.evaluate_bias import (
    normalize_for_model,
)

from data.transforms import to_common_image


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

CLASS_NAMES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


# ============================================================
# Load saved classifiers
# ============================================================

def load_classifiers(
    resnet,
    vit,
    clip,
    device,
):

    classifiers = {}

    resnet_classifier = LinearClassifier(
        input_dim=resnet.feature_dim,
        num_classes=10,
    ).to(device)

    vit_classifier = LinearClassifier(
        input_dim=vit.feature_dim,
        num_classes=10,
    ).to(device)

    clip_classifier = LinearClassifier(
        input_dim=clip.feature_dim,
        num_classes=10,
    ).to(device)

    resnet_classifier.load_state_dict(
        torch.load(
            PROJECT_ROOT
            / "results"
            / "resnet_classifier.pt",
            map_location=device,
        )
    )

    vit_classifier.load_state_dict(
        torch.load(
            PROJECT_ROOT
            / "results"
            / "vit_classifier.pt",
            map_location=device,
        )
    )

    clip_classifier.load_state_dict(
        torch.load(
            PROJECT_ROOT
            / "results"
            / "clip_classifier.pt",
            map_location=device,
        )
    )

    classifiers["resnet50"] = resnet_classifier
    classifiers["vit_b16"] = vit_classifier
    classifiers["clip"] = clip_classifier

    for classifier in classifiers.values():
        classifier.eval()

    return classifiers


# ============================================================
# Image conversion
# ============================================================

def image_to_tensor(image):

    image = to_common_image(
        image,
        IMAGE_SIZE
    )

    array = np.asarray(
        image
    )

    tensor = torch.from_numpy(
        array
    ).permute(2, 0, 1)

    return tensor


# ============================================================
# Predict with linear classifier
# ============================================================

@torch.no_grad()
def predict_linear_classifier(
    backbone,
    classifier,
    images,
    model_name,
    device,
):

    predictions = []

    backbone.eval()
    classifier.eval()

    for start in range(
        0,
        len(images),
        BATCH_SIZE,
    ):

        batch_images = images[
            start:start + BATCH_SIZE
        ]

        tensors = [
            image_to_tensor(image)
            for image in batch_images
        ]

        batch = torch.stack(
            tensors
        ).to(device)

        batch = normalize_for_model(
            batch,
            model_name
        )

        features = backbone(batch)

        logits = classifier(
            features
        )

        batch_predictions = (
            torch.argmax(
                logits,
                dim=1
            )
            .cpu()
            .numpy()
        )

        predictions.extend(
            batch_predictions.tolist()
        )

    return np.asarray(
        predictions
    )


# ============================================================
# CLIP zero-shot prediction
# ============================================================

@torch.no_grad()
def predict_zero_shot_clip(
    clip_backbone,
    images,
    device,
):

    clip_model = clip_backbone.model
    tokenizer = clip_backbone.tokenizer

    clip_model.eval()

    prompts = [
        f"a photo of a {class_name}."
        for class_name in CLASS_NAMES
    ]

    text_tokens = tokenizer(
        prompts
    ).to(device)

    text_features = (
        clip_model.encode_text(
            text_tokens
        )
    )

    text_features = (
        torch.nn.functional.normalize(
            text_features,
            dim=1
        )
    )

    predictions = []

    for start in range(
        0,
        len(images),
        BATCH_SIZE,
    ):

        batch_images = images[
            start:start + BATCH_SIZE
        ]

        tensors = [
            image_to_tensor(image)
            for image in batch_images
        ]

        batch = torch.stack(
            tensors
        ).to(device)

        batch = normalize_for_model(
            batch,
            "clip"
        )

        image_features = (
            clip_model.encode_image(
                batch
            )
        )

        image_features = (
            torch.nn.functional.normalize(
                image_features,
                dim=1
            )
        )

        logits = (
            clip_model.logit_scale.exp()
            * image_features
            @ text_features.T
        )

        batch_predictions = (
            torch.argmax(
                logits,
                dim=1
            )
            .cpu()
            .numpy()
        )

        predictions.extend(
            batch_predictions.tolist()
        )

    return np.asarray(
        predictions
    )


# ============================================================
# Evaluate shape / texture
# ============================================================

def classify_prediction(
    prediction,
    shape_class,
    style_class,
):

    shape_label = CLASS_NAMES.index(
        shape_class
    )

    style_label = CLASS_NAMES.index(
        style_class
    )

    if prediction == shape_label:
        return "shape"

    if prediction == style_label:
        return "texture"

    return "other"


# ============================================================
# Main evaluation
# ============================================================

def evaluate_cue_conflicts():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata_file = (
        PROJECT_ROOT
        / "results"
        / "cue_conflict_metadata.csv"
    )

    metadata = pd.read_csv(
        metadata_file
    )

    accepted_metadata = metadata[
        metadata["status"] == "accepted"
    ].copy()

    print(
        "\nAccepted conflicts:",
        len(accepted_metadata)
    )

    # --------------------------------------------------------
    # Load backbones
    # --------------------------------------------------------

    print("\nLoading backbones...")

    resnet = ResNet50Backbone().to(device)
    vit = ViTB16Backbone().to(device)
    clip = CLIPViTB32Backbone().to(device)

    models = {
        "resnet50": resnet,
        "vit_b16": vit,
        "clip": clip,
    }

    # --------------------------------------------------------
    # Load classifiers
    # --------------------------------------------------------

    classifiers = load_classifiers(
        resnet,
        vit,
        clip,
        device,
    )

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    images = []

    for _, row in accepted_metadata.iterrows():

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        images.append(image)

    # --------------------------------------------------------
    # Evaluate each model
    # --------------------------------------------------------

    all_results = []

    for model_name in [
        "resnet50",
        "vit_b16",
        "clip",
    ]:

        print(
            f"\nEvaluating {model_name}..."
        )

        predictions = (
            predict_linear_classifier(
                backbone=models[model_name],
                classifier=classifiers[model_name],
                images=images,
                model_name=model_name,
                device=device,
            )
        )

        for i, prediction in enumerate(
            predictions
        ):

            row = accepted_metadata.iloc[i]

            category = classify_prediction(
                prediction=int(prediction),
                shape_class=row["content_class"],
                style_class=row["style_class"],
            )

            all_results.append({
                "model": model_name,
                "content_class": row[
                    "content_class"
                ],
                "style_class": row[
                    "style_class"
                ],
                "prediction": CLASS_NAMES[
                    int(prediction)
                ],
                "cue_category": category,
                "image_path": row[
                    "image_path"
                ],
            })

    # --------------------------------------------------------
    # Zero-shot CLIP
    # --------------------------------------------------------

    print(
        "\nEvaluating CLIP zero-shot..."
    )

    predictions = predict_zero_shot_clip(
        clip,
        images,
        device,
    )

    for i, prediction in enumerate(
        predictions
    ):

        row = accepted_metadata.iloc[i]

        category = classify_prediction(
            prediction=int(prediction),
            shape_class=row["content_class"],
            style_class=row["style_class"],
        )

        all_results.append({
            "model": "clip_zero_shot",
            "content_class": row[
                "content_class"
            ],
            "style_class": row[
                "style_class"
            ],
            "prediction": CLASS_NAMES[
                int(prediction)
            ],
            "cue_category": category,
            "image_path": row[
                "image_path"
            ],
        })

    results = pd.DataFrame(
        all_results
    )

    # ========================================================
    # Calculate Shape Bias and Coverage
    # ========================================================

    summary = []

    total = len(
        accepted_metadata
    )

    for model_name in results[
        "model"
    ].unique():

        model_results = results[
            results["model"]
            == model_name
        ]

        n_shape = (
            model_results["cue_category"] == "shape"
        ).sum()

        n_texture = (
            model_results["cue_category"] == "texture"
        ).sum()

        n_other = (
            model_results["cue_category"] == "other"
        ).sum()

        shape_bias = (
            n_shape
            / (n_shape + n_texture)
            * 100
            if (n_shape + n_texture) > 0
            else 0.0
        )

        coverage = (
            (n_shape + n_texture)
            / total
            * 100
        )

        summary.append({
            "model": model_name,
            "Ntotal": total,
            "Nshape": int(n_shape),
            "Ntexture": int(n_texture),
            "Nother": int(n_other),
            "shape_bias_percent": shape_bias,
            "coverage_percent": coverage,
        })

    summary_df = pd.DataFrame(
        summary
    )

    # ========================================================
    # Save
    # ========================================================

    results_file = (
        PROJECT_ROOT
        / "results"
        / "cue_conflict_predictions.csv"
    )

    summary_file = (
        PROJECT_ROOT
        / "results"
        / "cue_conflict_summary.csv"
    )

    results.to_csv(
        results_file,
        index=False
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CUE-CONFLICT RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nPrediction-level results:"
    )

    print(results_file)

    print(
        "\nSummary:"
    )

    print(summary_file)


if __name__ == "__main__":
    evaluate_cue_conflicts()