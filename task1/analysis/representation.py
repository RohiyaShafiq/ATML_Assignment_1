import sys
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import STL10
from torchvision import transforms
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import open_clip

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs.config import (
    SEED,
    BATCH_SIZE,
    RESULTS_DIR)

from models.backbones import (
    ResNet50Backbone,
    ViTB16Backbone,
    CLIPViTB32Backbone)

from data.transforms import (
    to_common_image,
    grayscale,
    translate_image,
    patch_shuffle,
    IMAGENET_MEAN,
    IMAGENET_STD)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSLATION_PIXELS = 32
PATCH_SIZE = 4

TSNE_PERPLEXITY = 30
TSNE_ITERATIONS = 1000
print("Device:", DEVICE)


TEST_INDICES_FILE = (
    RESULTS_DIR / "test_indices.npy")

CUE_METADATA_FILE = (
    RESULTS_DIR / "cue_conflict_metadata.csv")

OUTPUT_DIR = (
    RESULTS_DIR / "representation")

FIGURES_DIR = (
    OUTPUT_DIR / "figures")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True)


imagenet_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD)])

# Dataset
class RepresentationDataset(Dataset):

    def __init__(
        self,
        dataset,
        indices,
        intervention,
        model_name,
        clip_preprocess=None,
        direction="right"):

        self.dataset = dataset
        self.indices = indices
        self.intervention = intervention
        self.model_name = model_name
        self.clip_preprocess = clip_preprocess
        self.direction = direction

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        image_index = int(self.indices[idx])
        image, label = self.dataset[image_index]

        image = to_common_image(image)
        if self.intervention == "clean":

            pass

        elif self.intervention == "grayscale":
            image = grayscale(image)

        elif self.intervention == "patch_shuffle":
            image = patch_shuffle(
                image,
                seed=SEED + image_index,
                patch_size=PATCH_SIZE)

        elif self.intervention == "translation":

            dx, dy = 0, 0
            if self.direction == "right":
                dx = TRANSLATION_PIXELS

            elif self.direction == "left":
                dx = -TRANSLATION_PIXELS

            elif self.direction == "down":
                dy = TRANSLATION_PIXELS

            elif self.direction == "up":
                dy = -TRANSLATION_PIXELS

            image = translate_image(image, dx, dy)

        else:
            raise ValueError(
                f"Unknown intervention: "
                f"{self.intervention}")

        if self.model_name in [
            "resnet50",
            "vit_b16"]:

            image = imagenet_transform(image)

        else:
            image = self.clip_preprocess(image)

        return image, label, image_index


# Feature extraction
def extract_features(backbone, dataset):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0)

    backbone.eval()

    features = []
    labels = []

    with torch.no_grad():

        for images, batch_labels, _ in loader:

            images = images.to(DEVICE)

            features.append(
                backbone(images)
                .cpu()
                .numpy())

            labels.append(
                batch_labels.numpy())

    return (
        np.concatenate(features),
        np.concatenate(labels))


# Cosine stability
def cosine_stability(clean, transformed):

    clean = normalize(clean)
    transformed = normalize(transformed)

    return np.sum(clean * transformed, axis=1)


# t-SNE
def make_tsne(clean, transformed, labels, model_name, intervention):

    features = np.concatenate([clean, transformed], axis=0)
    labels = np.concatenate([labels, labels], axis=0)

    conditions = np.array(
        ["Clean"] * len(clean)
        + ["Transformed"] * len(transformed))

    # Normalize before t-SNE
    features = normalize(features)

    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_ITERATIONS,
        random_state=SEED,
        init="pca",
        learning_rate="auto")

    embedding = tsne.fit_transform(features)

    plt.figure(figsize=(9, 7))

    # Ground-truth class = color
    # Condition = marker

    for class_id in range(10):

        for condition, marker in [("Clean", "o"),("Transformed", "x")]:

            mask = ((labels == class_id)
                & (conditions == condition))

            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                marker=marker,
                s=25,
                alpha=0.65,
                label=f"class {class_id} - {condition}")

    plt.title(f"{model_name} - {intervention}")

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7)

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / f"{model_name}_{intervention}_tsne.png")

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight")

    plt.close()

    print("Saved:", path)


# Load data
test_dataset = STL10(
    root=PROJECT_ROOT / "data" / "stl10",
    split="test",
    download=False)

test_indices = np.load(TEST_INDICES_FILE)


# CLIP preprocessing
_, clip_preprocess, _ = (
    open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
    )
)

# Load frozen backbones
backbones = {
    "resnet50": (ResNet50Backbone().to(DEVICE), None),
    "vit_b16": (ViTB16Backbone().to(DEVICE), None),
    "clip": (CLIPViTB32Backbone().to(DEVICE), clip_preprocess)}


# Run representation analysis
interventions = [
    "grayscale",
    "translation",
    "patch_shuffle"]

results = []


for model_name, (backbone, clip_transform) in backbones.items():

    print("MODEL:", model_name)

    # Clean representation
    clean_dataset = RepresentationDataset(
        test_dataset,
        test_indices,
        "clean",
        model_name,
        clip_transform)

    clean_features, labels = extract_features(
        backbone,
        clean_dataset)

    # Grayscale
    grayscale_dataset = RepresentationDataset(
        test_dataset,
        test_indices,
        "grayscale",
        model_name,
        clip_transform)

    grayscale_features, _ = extract_features(
        backbone,
        grayscale_dataset)

    cosine = cosine_stability(
        clean_features,
        grayscale_features)

    results.append({
        "model": model_name,
        "intervention": "grayscale",
        "cosine_stability": cosine.mean()})

    print(
        "Grayscale:",
        f"{cosine.mean():.4f}")

    make_tsne(
        clean_features,
        grayscale_features,
        labels,
        model_name,
        "grayscale")

    # Patch shuffle
    patch_dataset = RepresentationDataset(
        test_dataset,
        test_indices,
        "patch_shuffle",
        model_name,
        clip_transform)

    patch_features, _ = extract_features(
        backbone,
        patch_dataset)

    cosine = cosine_stability(
        clean_features,
        patch_features)

    results.append({
        "model": model_name,
        "intervention": "patch_shuffle",
        "cosine_stability": cosine.mean()})

    print(
        "Patch shuffle:",
        f"{cosine.mean():.4f}")

    make_tsne(
        clean_features,
        patch_features,
        labels,
        model_name,
        "patch_shuffle")

    # Translation
    translation_cosines = []

    # Average cosine stability over four directions
    for direction in ["right", "left", "down", "up"]:

        translation_dataset = (
            RepresentationDataset(
                test_dataset,
                test_indices,
                "translation",
                model_name,
                clip_transform,
                direction))

        translation_features, _ = (
            extract_features(
                backbone,
                translation_dataset))

        cosine = cosine_stability(
            clean_features,
            translation_features)

        translation_cosines.append(
            cosine.mean())

    mean_translation_cosine = np.mean(
        translation_cosines)

    results.append({
        "model": model_name,
        "intervention": "translation_32px",
        "cosine_stability":
            mean_translation_cosine})

    print(
        "Translation 32px:",
        f"{mean_translation_cosine:.4f}")

    # Use right translation for visualization
    translation_dataset = RepresentationDataset(
        test_dataset,
        test_indices,
        "translation",
        model_name,
        clip_transform,
        "right")

    translation_features, _ = (
        extract_features(
            backbone,
            translation_dataset))

    make_tsne(
        clean_features,
        translation_features,
        labels,
        model_name,
        "translation_32px")


# Cue-conflict representation
cue_metadata = pd.read_csv(
    CUE_METADATA_FILE)

cue_metadata = cue_metadata[
    cue_metadata["status"] == "accepted"].reset_index(drop=True)


class CueDataset(Dataset):

    def __init__(
        self,
        metadata,
        train_dataset,
        model_name,
        clip_preprocess=None):

        self.metadata = metadata
        self.train_dataset = train_dataset
        self.model_name = model_name
        self.clip_preprocess = clip_preprocess

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        row = self.metadata.iloc[idx]

        content_index = int(
            row["content_index"])

        clean_image, label = (
            self.train_dataset[
                content_index])

        conflict_image = Image.open(
            row["image_path"]).convert("RGB")

        clean_image = to_common_image(
            clean_image)

        conflict_image = to_common_image(
            conflict_image)

        if self.model_name in ["resnet50", "vit_b16"]:

            clean_image = imagenet_transform(clean_image)
            conflict_image = imagenet_transform(conflict_image)

        else:

            clean_image = (
                self.clip_preprocess(
                    clean_image))

            conflict_image = (
                self.clip_preprocess(
                    conflict_image))

        return (clean_image, conflict_image, label)


train_dataset = STL10(
    root=PROJECT_ROOT / "data" / "stl10",
    split="train",
    download=False)


for model_name, (backbone, clip_transform) in backbones.items():

    dataset = CueDataset(
        cue_metadata,
        train_dataset,
        model_name,
        clip_transform)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0)

    clean_features = []
    conflict_features = []
    labels = []

    backbone.eval()

    with torch.no_grad():

        for (
            clean_images,
            conflict_images,
            batch_labels) in loader:

            clean_images = (clean_images.to(DEVICE))
            conflict_images = (
                conflict_images.to(DEVICE))

            clean_features.append(
                backbone(clean_images)
                .cpu()
                .numpy())

            conflict_features.append(
                backbone(conflict_images)
                .cpu()
                .numpy())

            labels.append(batch_labels.numpy())

    clean_features = np.concatenate(clean_features)
    conflict_features = np.concatenate(conflict_features)
    labels = np.concatenate(labels)
    cosine = cosine_stability(clean_features, conflict_features)

    results.append({
        "model": model_name,
        "intervention": "cue_conflict",
        "cosine_stability": cosine.mean()})

    print(
        f"{model_name} Cue conflict:",
        f"{cosine.mean():.4f}")

    make_tsne(
        clean_features,
        conflict_features,
        labels,
        model_name,
        "cue_conflict")

results_df = pd.DataFrame(results)
output_file = (
    OUTPUT_DIR
    / "cosine_stability.csv")

results_df.to_csv(
    output_file,
    index=False)

print("REPRESENTATION ANALYSIS COMPLETE")
print(results_df.to_string(index=False))

print(
    "\nCosine results:",
    output_file)

print(
    "Figures:",
    FIGURES_DIR
)