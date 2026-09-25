import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import STL10
from torchvision import transforms
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs.config import (
    SEED,
    BATCH_SIZE,
    RESULTS_DIR)

from models.backbones import (
    ResNet50Backbone,
    ViTB16Backbone,
    CLIPViTB32Backbone,
    LinearClassifier)

from data.transforms import (
    to_common_image,
    translate_image,
    IMAGENET_MEAN,
    IMAGENET_STD)


torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Paths of the models and test images
TEST_INDICES_FILE = RESULTS_DIR / "test_indices.npy"

RESNET_WEIGHTS = RESULTS_DIR / "resnet_classifier.pt"
VIT_WEIGHTS = RESULTS_DIR / "vit_classifier.pt"
CLIP_WEIGHTS = RESULTS_DIR / "clip_classifier.pt"

OUTPUT_CSV = RESULTS_DIR / "translation_results.csv"

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# STL-10 dataset
class TranslationDataset(Dataset):

    def __init__(
        self,
        base_dataset,
        indices,
        displacement,
        direction):

        self.base_dataset = base_dataset
        self.indices = indices
        self.displacement = displacement
        self.direction = direction

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        original_index = int(self.indices[idx])
        image, label = self.base_dataset[original_index]
        image = to_common_image(image)

        dx = 0
        dy = 0

        if self.displacement > 0:

            if self.direction == "right":
                dx = self.displacement

            elif self.direction == "left":
                dx = -self.displacement

            elif self.direction == "down":
                dy = self.displacement

            elif self.direction == "up":
                dy = -self.displacement

        image = translate_image(image, dx=dx, dy=dy)

        return image, label, original_index


# Model-specific preprocessing
imagenet_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD)])


class ImageTransformDataset(Dataset):

    def __init__(
        self,
        translation_dataset,
        model_name):
        self.dataset = translation_dataset
        self.model_name = model_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image, label, original_index = self.dataset[idx]

        if self.model_name in ["resnet50", "vit_b16"]:
            image = imagenet_transform(image)

        elif self.model_name == "clip":
            # OpenCLIP preprocessing
            image = image

        return image, label, original_index


# CLIP preprocessing
def get_clip_preprocess(clip_model):

    import open_clip

    _, preprocess, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai")

    return preprocess


# Prediction helper
def get_predictions(
    backbone,
    classifier,
    dataloader,
    model_name,
):

    backbone.eval()
    classifier.eval()

    all_predictions = []
    all_labels = []
    all_indices = []

    with torch.no_grad():

        for images, labels, indices in dataloader:

            if model_name == "clip":
                # Images may still be PIL images
                preprocess = get_clip_preprocess(backbone.model)

                images = torch.stack([
                    preprocess(image)
                    for image in images])

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            features = backbone(images)

            logits = classifier(features)

            predictions = logits.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy())

            all_labels.extend(
                labels.cpu().numpy())

            all_indices.extend(
                indices.numpy())

    return (
        np.array(all_predictions),
        np.array(all_labels),
        np.array(all_indices))


# Better CLIP preprocessing dataset
class ModelTranslationDataset(Dataset):

    def __init__(
        self,
        base_dataset,
        indices,
        displacement,
        direction,
        model_name,
        clip_preprocess=None):

        self.base_dataset = base_dataset
        self.indices = indices
        self.displacement = displacement
        self.direction = direction
        self.model_name = model_name
        self.clip_preprocess = clip_preprocess

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        original_index = int(self.indices[idx])
        image, label = self.base_dataset[original_index]
        image = to_common_image(image)

        dx = 0
        dy = 0

        if self.displacement > 0:

            if self.direction == "right":
                dx = self.displacement

            elif self.direction == "left":
                dx = -self.displacement

            elif self.direction == "down":
                dy = self.displacement

            elif self.direction == "up":
                dy = -self.displacement

        image = translate_image(
            image,
            dx=dx,
            dy=dy,
        )

        if self.model_name in ["resnet50", "vit_b16"]:
            image = imagenet_transform(image)

        elif self.model_name == "clip":
            image = self.clip_preprocess(image)

        return image, label, original_index


# Load models
print("\nLoading models...")

resnet = ResNet50Backbone().to(DEVICE)
vit = ViTB16Backbone().to(DEVICE)
clip = CLIPViTB32Backbone().to(DEVICE)

# Load classifier heads
resnet_classifier = LinearClassifier(resnet.feature_dim, 10)
vit_classifier = LinearClassifier(vit.feature_dim, 10)
clip_classifier = LinearClassifier(clip.feature_dim, 10)

resnet_classifier.load_state_dict(
    torch.load(
        RESNET_WEIGHTS,
        map_location=DEVICE,
    )
)

vit_classifier.load_state_dict(
    torch.load(
        VIT_WEIGHTS,
        map_location=DEVICE,
    )
)

clip_classifier.load_state_dict(
    torch.load(
        CLIP_WEIGHTS,
        map_location=DEVICE,
    )
)


resnet_classifier = resnet_classifier.to(DEVICE)
vit_classifier = vit_classifier.to(DEVICE)
clip_classifier = clip_classifier.to(DEVICE)


models = {
    "resnet50": (resnet, resnet_classifier),
    "vit_b16": (vit, vit_classifier),
    "clip": (clip, clip_classifier)}


# Load STL-10 test set
base_dataset = STL10(root=PROJECT_ROOT / "data" / "stl10", split="test", download=False)
test_indices = np.load(TEST_INDICES_FILE)

print("Test subset size:", len(test_indices))


# CLIP preprocessing
import open_clip

_, clip_preprocess, _ = (
    open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
    )
)


# Directions and displacements
DISPLACEMENTS = [0, 8, 16, 32]
DIRECTIONS = ["right","left","down","up"]

# Evaluate translation
results = []

for model_name, (backbone, classifier) in models.items():

    print("MODEL:", model_name)

    # Clean predictions
    clean_dataset = ModelTranslationDataset(
        base_dataset=base_dataset,
        indices=test_indices,
        displacement=0,
        direction="right",
        model_name=model_name,
        clip_preprocess=clip_preprocess
        if model_name == "clip"
        else None)

    clean_loader = DataLoader(
        clean_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0)

    backbone.eval()
    classifier.eval()

    clean_predictions = {}

    with torch.no_grad():

        for images, labels, indices in clean_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            features = backbone(images)

            logits = classifier(features)

            predictions = logits.argmax(dim=1)

            for idx, pred in zip(
                indices.numpy(),
                predictions.cpu().numpy(),
            ):
                clean_predictions[int(idx)] = int(pred)

    clean_accuracy = np.mean([
        clean_predictions[int(idx)]
        == int(base_dataset.labels[int(idx)])
        for idx in test_indices
    ])

    print(
        f"Clean accuracy: "
        f"{clean_accuracy:.4f}")

    # Translation experiments
    for displacement in DISPLACEMENTS:

        direction_accuracies = []
        direction_consistencies = []

        for direction in DIRECTIONS:

            print(
                f"{model_name} | "
                f"displacement={displacement} | "
                f"direction={direction}")

            dataset = ModelTranslationDataset(
                base_dataset=base_dataset,
                indices=test_indices,
                displacement=displacement,
                direction=direction,
                model_name=model_name,
                clip_preprocess=clip_preprocess
                if model_name == "clip"
                else None)

            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0)

            translated_predictions = {}

            translated_labels = {}

            with torch.no_grad():

                for images, labels, indices in loader:

                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)

                    features = backbone(images)

                    logits = classifier(features)

                    predictions = logits.argmax(
                        dim=1
                    )

                    for idx, pred, label in zip(
                        indices.cpu().numpy(),
                        predictions.cpu().numpy(),
                        labels.cpu().numpy()):

                        translated_predictions[
                            int(idx)
                        ] = int(pred)

                        translated_labels[
                            int(idx)
                        ] = int(label)

            # Accuracy
            accuracy = np.mean([
                translated_predictions[int(idx)]
                == translated_labels[int(idx)]
                for idx in test_indices
            ])

            # Prediction consistency
            consistency = np.mean([
                translated_predictions[int(idx)]
                == clean_predictions[int(idx)]
                for idx in test_indices])

            direction_accuracies.append(accuracy)
            direction_consistencies.append(consistency)

            results.append({
                "model": model_name,
                "displacement": displacement,
                "direction": direction,
                "accuracy": accuracy,
                "consistency": consistency})

        # Average across four directions
        mean_accuracy = np.mean(direction_accuracies)
        mean_consistency = np.mean(direction_consistencies)

        results.append({
            "model": model_name,
            "displacement": displacement,
            "direction": "mean",
            "accuracy": mean_accuracy,
            "consistency": mean_consistency})

        print(
            f"Mean | displacement={displacement} | "
            f"Accuracy={mean_accuracy:.4f} | "
            f"Consistency={mean_consistency:.4f}")


# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)

print("\nResults saved to:")
print(OUTPUT_CSV)


# Plot 1: Accuracy vs displacement
plt.figure(figsize=(8, 5))

for model_name in models.keys():

    model_df = results_df[
        (results_df["model"] == model_name)
        & (results_df["direction"] == "mean")
    ]

    plt.plot(
        model_df["displacement"],
        model_df["accuracy"],
        marker="o",
        label=model_name,
    )

plt.xlabel("Displacement (pixels)")
plt.ylabel("Top-1 Accuracy")
plt.title("Translation Robustness: Accuracy")
plt.xticks(DISPLACEMENTS)
plt.ylim(0, 1)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

accuracy_fig = (
    FIGURES_DIR
    / "translation_accuracy.png")

plt.savefig(
    accuracy_fig,
    dpi=300,
    bbox_inches="tight")

plt.show()


# Plot 2: Prediction consistency vs displacement
plt.figure(figsize=(8, 5))

for model_name in models.keys():

    model_df = results_df[
        (results_df["model"] == model_name)
        & (results_df["direction"] == "mean")]

    plt.plot(
        model_df["displacement"],
        model_df["consistency"],
        marker="o",
        label=model_name)

plt.xlabel("Displacement (pixels)")
plt.ylabel("Prediction Consistency")
plt.title("Translation Robustness: Prediction Consistency")

plt.xticks(DISPLACEMENTS)
plt.ylim(0, 1)

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

consistency_fig = (
    FIGURES_DIR
    / "translation_consistency.png"
)

plt.savefig(
    consistency_fig,
    dpi=300,
    bbox_inches="tight")

plt.show()


# Final summary
summary_df = results_df[results_df["direction"] == "mean"].copy()
print("TRANSLATION ROBUSTNESS RESULTS")
print(
    summary_df[
        [
            "model",
            "displacement",
            "accuracy",
            "consistency"
        ]
    ].to_string(index=False)
)

print("\nFigures saved to:")
print(accuracy_fig)
print(consistency_fig)