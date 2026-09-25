import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import STL10
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs.config import (
    SEED,
    BATCH_SIZE,
    RESULTS_DIR,
)

from models.backbones import (
    ResNet50Backbone,
    ViTB16Backbone,
    CLIPViTB32Backbone,
    LinearClassifier,
)

from data.transforms import (
    to_common_image,
    patch_shuffle,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)

TEST_INDICES_FILE = (
    RESULTS_DIR / "test_indices.npy"
)

RESNET_WEIGHTS = (
    RESULTS_DIR / "resnet_classifier.pt"
)

VIT_WEIGHTS = (
    RESULTS_DIR / "vit_classifier.pt"
)

CLIP_WEIGHTS = (
    RESULTS_DIR / "clip_classifier.pt"
)

OUTPUT_CSV = (
    RESULTS_DIR / "patch_shuffle_results.csv"
)



# Image preprocessing
imagenet_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# Dataset
class PatchShuffleDataset(Dataset):

    def __init__(
        self,
        base_dataset,
        indices,
        model_name,
        clip_preprocess=None,
        shuffle_patches=False,
        seed=6304,
    ):

        self.base_dataset = base_dataset
        self.indices = indices
        self.model_name = model_name
        self.clip_preprocess = clip_preprocess
        self.shuffle_patches = shuffle_patches
        self.seed = seed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        original_index = int(self.indices[idx])
        image, label = self.base_dataset[original_index]
        image = to_common_image(image)

        if self.shuffle_patches:

            # Use the SAME permutation for the
            # same image regardless of model.
            #
            # The image-specific seed guarantees
            # deterministic shuffling.

            image = patch_shuffle(
                image,
                seed=self.seed + original_index,
                patch_size=4)

        if self.model_name in [
            "resnet50",
            "vit_b16"]:

            image = imagenet_transform(image)

        elif self.model_name == "clip":
            image = self.clip_preprocess(image)

        return (image, label, original_index)


print("\nLoading models...")
resnet = ResNet50Backbone().to(DEVICE)
vit = ViTB16Backbone().to(DEVICE)
clip = CLIPViTB32Backbone().to(DEVICE)

resnet_classifier = LinearClassifier(resnet.feature_dim, 10)
vit_classifier = LinearClassifier(vit.feature_dim, 10)
clip_classifier = LinearClassifier(clip.feature_dim, 10)

resnet_classifier.load_state_dict(
    torch.load(
        RESNET_WEIGHTS,
        map_location=DEVICE))

vit_classifier.load_state_dict(
    torch.load(
        VIT_WEIGHTS,
        map_location=DEVICE))

clip_classifier.load_state_dict(
    torch.load(
        CLIP_WEIGHTS,
        map_location=DEVICE))

resnet_classifier = (resnet_classifier.to(DEVICE))
vit_classifier = (vit_classifier.to(DEVICE))
clip_classifier = (clip_classifier.to(DEVICE))

models = {
    "resnet50": (
        resnet,
        resnet_classifier),

    "vit_b16": (
        vit,
        vit_classifier),

    "clip": (
        clip,
        clip_classifier)}

import open_clip

_, clip_preprocess, _ = (
    open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai"))

base_dataset = STL10(
    root=PROJECT_ROOT / "data" / "stl10",
    split="test",
    download=False)

test_indices = np.load(TEST_INDICES_FILE)

print(
    "Test subset size:",
    len(test_indices))

def get_predictions(
    backbone,
    classifier,
    dataloader):

    backbone.eval()
    classifier.eval()

    predictions = {}
    labels_dict = {}

    with torch.no_grad():

        for (
            images,
            labels,
            indices,
        ) in dataloader:

            images = images.to(DEVICE)

            features = backbone(images)

            logits = classifier(features)

            preds = logits.argmax(dim=1)

            for (
                idx,
                pred,
                label,
            ) in zip(
                indices.numpy(),
                preds.cpu().numpy(),
                labels.numpy(),
            ):

                predictions[int(idx)] = int(pred)

                labels_dict[int(idx)] = int(label)

    return predictions, labels_dict

results = []

for model_name, (
    backbone,
    classifier,
) in models.items():

    print("MODEL:", model_name)
    clean_dataset = PatchShuffleDataset(
        base_dataset=base_dataset,
        indices=test_indices,
        model_name=model_name,
        clip_preprocess=(
            clip_preprocess
            if model_name == "clip"
            else None
        ),
        shuffle_patches=False,
        seed=SEED,
    )

    clean_loader = DataLoader(
        clean_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    clean_predictions, labels = (
        get_predictions(
            backbone,
            classifier,
            clean_loader,
        )
    )

    clean_accuracy = np.mean([
        clean_predictions[int(idx)]
        == labels[int(idx)]
        for idx in test_indices
    ])

    print(
        f"Clean accuracy: "
        f"{clean_accuracy:.4f}"
    )

    shuffled_dataset = PatchShuffleDataset(
        base_dataset=base_dataset,
        indices=test_indices,
        model_name=model_name,
        clip_preprocess=(
            clip_preprocess
            if model_name == "clip"
            else None
        ),
        shuffle_patches=True,
        seed=SEED,
    )

    shuffled_loader = DataLoader(
        shuffled_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    shuffled_predictions, shuffled_labels = (
        get_predictions(
            backbone,
            classifier,
            shuffled_loader,
        )
    )

    shuffled_accuracy = np.mean([
        shuffled_predictions[int(idx)]
        == shuffled_labels[int(idx)]
        for idx in test_indices
    ])

    accuracy_drop = (
        clean_accuracy
        - shuffled_accuracy
    )

    consistency = np.mean([
        shuffled_predictions[int(idx)]
        == clean_predictions[int(idx)]
        for idx in test_indices
    ])

    results.append({
        "model": model_name,
        "clean_accuracy": clean_accuracy,
        "shuffled_accuracy": shuffled_accuracy,
        "accuracy_drop": accuracy_drop,
        "prediction_consistency": consistency,
    })


    print(
        f"Shuffled accuracy: "
        f"{shuffled_accuracy:.4f}"
    )

    print(
        f"Accuracy drop: "
        f"{accuracy_drop:.4f}"
    )

    print(
        f"Prediction consistency: "
        f"{consistency:.4f}"
    )


results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_CSV,
    index=False,
)

print("\n" + "=" * 70)
print("PATCH SHUFFLE RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nResults saved to:")
print(OUTPUT_CSV)