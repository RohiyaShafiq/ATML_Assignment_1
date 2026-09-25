import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torchvision.datasets import STL10
from torch.utils.data import TensorDataset, DataLoader
# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Imports
# ============================================================

from configs.config import (
    SEED,
    BATCH_SIZE,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    WEIGHT_DECAY,
)

from models.backbones import (
    ResNet50Backbone,
    ViTB16Backbone,
    CLIPViTB32Backbone,
    LinearClassifier,
    train_linear_classifier,
)

from analysis.evaluate_bias import (
    run_clean_baseline,
    extract_features,
    evaluate_color_intervention,
    load_clean_test_subset,
)

from data.transforms import (
    grayscale,
    hue_rotation,
)

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=SEED):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_feature_loader(features, labels, batch_size, shuffle):
    features = torch.as_tensor(features, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)

    dataset = TensorDataset(features, labels)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


# PART 2 - COLOR BIAS EXPERIMENT
def run_part2_color_bias(
    models,
    classifiers,
    device
):
    print("PART 2 - COLOR BIAS EXPERIMENT")

    # Load the EXACT same 500 test images
    images, labels, image_ids = load_clean_test_subset()

    # Grayscale
    grayscale_results = evaluate_color_intervention(
        models=models,
        classifiers=classifiers,
        images=images,
        labels=labels,
        device=device,
        intervention_name="grayscale",
        transform_function=grayscale,
    )

    # 2. Fixed 60-degree hue rotation
    hue_results = evaluate_color_intervention(
        models=models,
        classifiers=classifiers,
        images=images,
        labels=labels,
        device=device,
        intervention_name="hue_rotation_60",
        transform_function=lambda image: hue_rotation(
            image,
            degrees=60))

    # Combine results
    color_results = pd.concat(
        [
            grayscale_results,
            hue_results],ignore_index=True)

    print("\n")
    print("FINAL COLOR BIAS RESULTS")

    print(
        color_results.to_string(index=False))

    # Save results
    color_results_file = (
        PROJECT_ROOT
        / "results"
        / "color_bias_results.csv")

    color_results.to_csv(
        color_results_file,
        index=False)

    print(
        f"\nColor results saved to:\n"
        f"{color_results_file}")

def load_trained_classifiers(
    resnet,
    vit,
    clip,
    device
):

    resnet_classifier = LinearClassifier(
        input_dim=resnet.feature_dim,
        num_classes=10
    ).to(device)

    vit_classifier = LinearClassifier(
        input_dim=vit.feature_dim,
        num_classes=10
    ).to(device)

    clip_classifier = LinearClassifier(
        input_dim=clip.feature_dim,
        num_classes=10
    ).to(device)

    # Load saved weights
    resnet_classifier.load_state_dict(
        torch.load(
            PROJECT_ROOT / "results" / "resnet_classifier.pt",
            map_location=device
        )
    )

    vit_classifier.load_state_dict(
        torch.load(
            PROJECT_ROOT / "results" / "vit_classifier.pt",
            map_location=device
        )
    )

    clip_classifier.load_state_dict(
        torch.load(
            PROJECT_ROOT / "results" / "clip_classifier.pt",
            map_location=device
        )
    )

    resnet_classifier.eval()
    vit_classifier.eval()
    clip_classifier.eval()

    return {
        "resnet50": resnet_classifier,
        "vit_b16": vit_classifier,
        "clip": clip_classifier,
    }

def main():
  print("TASK 1 - PART 2: COLOR BIAS EXPERIMENT")

      set_seed(SEED)

      print("Seed:", SEED)

      device = torch.device(
          "cuda" if torch.cuda.is_available()
          else "cpu"
      )

      print("Device:", device)

      # ========================================================
      # 1. Load pretrained frozen backbones
      # ========================================================

      print("\nLoading pretrained frozen backbones...")

      print("\nLoading ResNet-50...")
      resnet = ResNet50Backbone().to(device)

      print("\nLoading ViT-B/16...")
      vit = ViTB16Backbone().to(device)

      print("\nLoading OpenCLIP ViT-B/32...")
      clip = CLIPViTB32Backbone().to(device)

      models = {
          "resnet50": resnet,
          "vit_b16": vit,
          "clip": clip,
      }

      # ========================================================
      # 2. Load saved classifier weights
      # ========================================================

      print("\nLoading saved classifier weights...")

      classifiers = load_trained_classifiers(
          resnet=resnet,
          vit=vit,
          clip=clip,
          device=device,
      )

      print("Classifier weights loaded successfully.")

      # ========================================================
      # 3. Run Part 2
      # ========================================================

      run_part2_color_bias(
          models=models,
          classifiers=classifiers,
          device=device
      )

      print("\nPART 2 COMPLETED.")

if __name__ == "__main__":
    main()