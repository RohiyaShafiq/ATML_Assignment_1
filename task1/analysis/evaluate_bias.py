from pathlib import Path
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs.config import (
    SEED,
    BATCH_SIZE,
    RESULTS_DIR,
    SUBSET_FILE,
    IMAGE_SIZE,
)

from data.transforms import (
    to_common_image,
    grayscale,
    hue_rotation,
)

# same random values whenever we run the code
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Model-specific normalization
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

# apply normalization on the images for each model
def normalize_for_model(images, model_name):

    images = images.float() / 255.0
    if model_name in ["resnet50", "vit_b16"]:
        mean = IMAGENET_MEAN.to(images.device)
        std = IMAGENET_STD.to(images.device)

    elif model_name == "clip":
        mean = CLIP_MEAN.to(images.device)
        std = CLIP_STD.to(images.device)

    else:
        raise ValueError(
            f"Unknown model: {model_name}")

    return (images - mean) / std


# Convert PIL image to tensor
def image_to_tensor(image):

    image = to_common_image(image, IMAGE_SIZE)
    array = np.asarray(image)
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor

# make batches of the images
def make_image_batch(images):

    tensors = [image_to_tensor(image) for image in images]
    return torch.stack(tensors)


# Load the fixed 500-image test subset
def load_clean_test_subset():
    from torchvision.datasets import STL10

    dataset = STL10(root=str(PROJECT_ROOT / "data"), split="test", download=True,)
    subset_df = pd.read_csv(SUBSET_FILE)

    images = []
    labels = []
    image_ids = []

    for _, row in subset_df.iterrows():

        test_index = int(row["test_index"])
        image, label = dataset[test_index]
        image = to_common_image(image, IMAGE_SIZE)

        images.append(image)
        labels.append(int(label))
        image_ids.append(test_index)

    return images, np.array(labels), image_ids


# Extract frozen backbone features
@torch.no_grad()
def extract_features(backbone, images, model_name, device,):

    backbone.eval()

    all_features = []

    for start in range(0, len(images), BATCH_SIZE):

        batch_images = images[start:start + BATCH_SIZE]
        batch = make_image_batch(batch_images).to(device)
        batch = normalize_for_model(batch, model_name)

        features = backbone(batch)
        if model_name == "clip":
            features = F.normalize(features, dim=1)

        all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


# Evaluate a trained linear classifier
@torch.no_grad()
def evaluate_classifier(classifier, features, labels, device,):

    classifier.eval()

    dataset = TensorDataset(features)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,)

    all_predictions = []
    all_probabilities = []

    for (batch_features,) in loader:

        batch_features = (batch_features.to(device))
        logits = classifier(batch_features)
        probabilities = F.softmax(logits, dim=1)

        predictions = torch.argmax(probabilities, dim=1)
        all_predictions.append(predictions.cpu())
        all_probabilities.append(probabilities.cpu())

    predictions = torch.cat(all_predictions)
    probabilities = torch.cat(all_probabilities)

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    accuracy = accuracy_score(labels_tensor.numpy(), predictions.numpy())

    macro_f1 = f1_score(
        labels_tensor.numpy(),
        predictions.numpy(),
        average="macro",
        zero_division=0)

    mean_max_confidence = (
        probabilities
        .max(dim=1)
        .values
        .mean()
        .item())

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "mean_max_confidence": mean_max_confidence}


# Zero-shot CLIP
@torch.no_grad()
def evaluate_zero_shot_clip(clip_backbone, images, labels, device,):

    clip_model = clip_backbone.model
    tokenizer = clip_backbone.tokenizer

    clip_model.eval()

    class_names = [
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

    # Required fixed prompt
    prompts = [
        f"a photo of a {class_name}."
        for class_name in class_names]

    text_tokens = tokenizer(prompts).to(device)

    # Encode text
    text_features = (clip_model.encode_text(text_tokens))

    # Normalize text embeddings
    text_features = F.normalize(text_features, dim=1)

    all_predictions = []
    all_probabilities = []

    for start in range(0, len(images), BATCH_SIZE):

        batch_images = images[start:start + BATCH_SIZE]
        batch = make_image_batch(batch_images).to(device)

        # CLIP normalization
        batch = normalize_for_model(batch,"clip")

        # Encode images
        image_features = (clip_model.encode_image(batch))

        # Normalize image embeddings
        image_features = F.normalize(image_features, dim=1)

        # Scaled class similarities
        logits = (
            clip_model.logit_scale.exp()
            * image_features
            @ text_features.T)

        # Required confidence calculation
        probabilities = F.softmax(logits, dim=1)

        predictions = torch.argmax(probabilities, dim=1)

        all_predictions.append(predictions.cpu())

        all_probabilities.append(probabilities.cpu())

    predictions = torch.cat(all_predictions)

    probabilities = torch.cat(all_probabilities)

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    accuracy = accuracy_score(labels_tensor.numpy(), predictions.numpy())

    macro_f1 = f1_score(
        labels_tensor.numpy(),
        predictions.numpy(),
        average="macro",
        zero_division=0)

    mean_max_confidence = (probabilities.max(dim=1).values.mean().item())

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "mean_max_confidence": mean_max_confidence,
    }

# Part 1
def run_clean_baseline(models, classifiers, device):

    set_seed(SEED)

    print("CLEAN BASELINE EVALUATION")
    
    # Load the same fixed 500 test images
    images, labels, image_ids = (load_clean_test_subset())
    print(f"\nNumber of test images: {len(images)}")

    # evaluate the results of 3 classifiers
    results = []

    for model_name in ["resnet50", "vit_b16", "clip"]:

        print(f"\nEvaluating {model_name}...")

        features = extract_features(
            backbone=models[model_name],
            images=images,
            model_name=model_name,
            device=device)

        metrics = evaluate_classifier(
            classifier=classifiers[model_name],
            features=features,
            labels=labels,
            device=device)

        results.append({
            "model": model_name,
            "evaluation": "clean",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "mean_max_confidence": (
                metrics["mean_max_confidence"])})

    # Evaluate zero-shot CLIP
    print("\nEvaluating zero-shot CLIP...")

    zero_shot_metrics = (
        evaluate_zero_shot_clip(
            clip_backbone=models["clip"],
            images=images,
            labels=labels,
            device=device))

    results.append({
        "model": "clip_zero_shot",
        "evaluation": "clean",
        "accuracy": zero_shot_metrics["accuracy"],
        "macro_f1": zero_shot_metrics["macro_f1"],
        "mean_max_confidence": (
            zero_shot_metrics["mean_max_confidence"])})

    # Create results table
    results_df = pd.DataFrame(results)
    print("CLEAN BASELINE RESULTS")
    print(results_df.to_string(index=False))

    # Save results
    output_file = (RESULTS_DIR / "clean_baseline_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results to:\n{output_file}")

    return results_df


# Part 2 - Color Bias Experiment
@torch.no_grad()
def evaluate_color_intervention(
    models,
    classifiers,
    images,
    labels,
    device,
    intervention_name,
    transform_function,
):
    """
    Evaluate a color intervention.

    Interventions:
        - grayscale
        - fixed hue rotation

    Metrics:
        - transformed accuracy
        - accuracy change relative to clean
        - prediction consistency relative to clean
    """

    print("\n" + "=" * 70)
    print(f"COLOR INTERVENTION: {intervention_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Create transformed versions of the SAME 500 images
    # --------------------------------------------------------

    transformed_images = [
        transform_function(image)
        for image in images
    ]

    results = []

    # --------------------------------------------------------
    # Evaluate the three trained classifier heads
    # --------------------------------------------------------

    for model_name in [
        "resnet50",
        "vit_b16",
        "clip"
    ]:

        print(
            f"\nEvaluating {model_name} "
            f"under {intervention_name}..."
        )

        # ====================================================
        # Clean predictions
        # ====================================================

        clean_features = extract_features(
            backbone=models[model_name],
            images=images,
            model_name=model_name,
            device=device
        )

        clean_metrics = evaluate_classifier(
            classifier=classifiers[model_name],
            features=clean_features,
            labels=labels,
            device=device
        )

        # Get clean predictions
        clean_logits = classifiers[model_name](
            clean_features.to(device)
        )

        clean_predictions = (
            torch.argmax(
                clean_logits,
                dim=1
            )
            .cpu()
        )

        # ====================================================
        # Transformed predictions
        # ====================================================

        transformed_features = extract_features(
            backbone=models[model_name],
            images=transformed_images,
            model_name=model_name,
            device=device
        )

        transformed_metrics = evaluate_classifier(
            classifier=classifiers[model_name],
            features=transformed_features,
            labels=labels,
            device=device
        )

        transformed_logits = classifiers[model_name](
            transformed_features.to(device)
        )

        transformed_predictions = (
            torch.argmax(
                transformed_logits,
                dim=1
            )
            .cpu()
        )

        # ====================================================
        # Prediction consistency
        # ====================================================

        prediction_consistency = (
            clean_predictions ==
            transformed_predictions
        ).float().mean().item()

        # ====================================================
        # Accuracy change
        # ====================================================

        accuracy_change = (
            transformed_metrics["accuracy"]
            - clean_metrics["accuracy"]
        )

        results.append({
            "model": model_name,
            "intervention": intervention_name,
            "clean_accuracy": (
                clean_metrics["accuracy"]
            ),
            "transformed_accuracy": (
                transformed_metrics["accuracy"]
            ),
            "accuracy_change": accuracy_change,
            "prediction_consistency": (
                prediction_consistency
            ),
        })

    # --------------------------------------------------------
    # Zero-shot CLIP
    # --------------------------------------------------------

    print(
        f"\nEvaluating zero-shot CLIP "
        f"under {intervention_name}..."
    )

    # Clean zero-shot CLIP predictions
    class_names = [
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

    prompts = [
        f"a photo of a {class_name}."
        for class_name in class_names
    ]

    clip_model = models["clip"].model
    tokenizer = models["clip"].tokenizer

    clip_model.eval()

    text_tokens = tokenizer(
        prompts
    ).to(device)

    text_features = clip_model.encode_text(
        text_tokens
    )

    text_features = F.normalize(
        text_features,
        dim=1
    )

    def get_zero_shot_predictions(image_list):

        all_predictions = []

        for start in range(
            0,
            len(image_list),
            BATCH_SIZE
        ):

            batch_images = image_list[
                start:start + BATCH_SIZE
            ]

            batch = make_image_batch(
                batch_images
            ).to(device)

            batch = normalize_for_model(
                batch,
                "clip"
            )

            image_features = (
                clip_model.encode_image(batch)
            )

            image_features = F.normalize(
                image_features,
                dim=1
            )

            logits = (
                clip_model.logit_scale.exp()
                * image_features
                @ text_features.T
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.append(
                predictions.cpu()
            )

        return torch.cat(
            all_predictions
        )

    # Clean predictions
    clean_zero_shot_predictions = (
        get_zero_shot_predictions(images)
    )

    # Transformed predictions
    transformed_zero_shot_predictions = (
        get_zero_shot_predictions(
            transformed_images
        )
    )

    labels_tensor = torch.tensor(
        labels,
        dtype=torch.long
    )

    clean_zero_shot_accuracy = accuracy_score(
        labels_tensor.numpy(),
        clean_zero_shot_predictions.numpy()
    )

    transformed_zero_shot_accuracy = (
        accuracy_score(
            labels_tensor.numpy(),
            transformed_zero_shot_predictions.numpy()
        )
    )

    zero_shot_accuracy_change = (
        transformed_zero_shot_accuracy
        - clean_zero_shot_accuracy
    )

    zero_shot_consistency = (
        clean_zero_shot_predictions ==
        transformed_zero_shot_predictions
    ).float().mean().item()

    results.append({
        "model": "clip_zero_shot",
        "intervention": intervention_name,
        "clean_accuracy": (
            clean_zero_shot_accuracy
        ),
        "transformed_accuracy": (
            transformed_zero_shot_accuracy
        ),
        "accuracy_change": (
            zero_shot_accuracy_change
        ),
        "prediction_consistency": (
            zero_shot_consistency
        ),
    })

    # --------------------------------------------------------
    # Results table
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    print(
        f"\n{intervention_name.upper()} RESULTS"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    return results_df