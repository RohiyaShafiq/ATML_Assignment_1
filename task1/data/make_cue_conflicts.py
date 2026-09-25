import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets import STL10


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

SEED = 6304

IMAGE_SIZE = 224

STYLE_ALPHA = 1

TARGET_CONFLICTS = 200

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "cue_conflicts")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True)

# STL-10 classes
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

# Five unordered class pairs
CLASS_PAIRS = [
    ("airplane", "bird"),
    ("car", "truck"),
    ("cat", "dog"),
    ("deer", "horse"),
    ("monkey", "ship")]


# Visual rejection rule
def visually_valid_conflict(content_image, stylized_image):

    if stylized_image.mode != "RGB":
        return False, "not_rgb"

    if stylized_image.size != (IMAGE_SIZE, IMAGE_SIZE):
        return False, "wrong_size"

    arr = np.asarray(
        stylized_image
    ).astype(np.float32)

    # Completely invalid output
    if not np.isfinite(arr).all():
        return False, "non_finite"

    # Almost uniform image
    if arr.std() < 8.0:
        return False, "near_uniform"

    # Severe clipping
    clipped_low = np.mean(arr <= 1)
    clipped_high = np.mean(arr >= 254)

    if clipped_low > 0.30:
        return False, "excessive_black"

    if clipped_high > 0.30:
        return False, "excessive_white"

    return True, "accepted"

# Get image indices by class
def build_class_indices(dataset):

    class_indices = {
        class_name: []
        for class_name in CLASS_NAMES
    }

    for index, label in enumerate(dataset.labels):

        class_name = CLASS_NAMES[int(label)]

        class_indices[class_name].append(index)

    return class_indices


# Generate conflicts
def generate_conflicts():

    random.seed(SEED)
    np.random.seed(SEED)

    # Load STL-10 training partition
    dataset = STL10(
        root=PROJECT_ROOT / "data" / "stl10",
        split="train",
        download=False,)

    print(
        "STL-10 training images:",
        len(dataset))

    # Build class index lookup
    class_indices = build_class_indices(dataset)

    # Load AdaIN
    import torch

    from models.adain import AdaINStyleTransfer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    adain = AdaINStyleTransfer(device=device)

    # Number per direction
    directions = []

    for class_a, class_b in CLASS_PAIRS:

        directions.append((class_a, class_b))
        directions.append((class_b, class_a))

    conflicts_per_direction = (
        TARGET_CONFLICTS
        // len(directions))

    print("Directions:", len(directions))

    print(
        "Target conflicts per direction:",
        conflicts_per_direction)

    records = []

    accepted_count = 0
    rejected_count = 0

    # Generate
    for shape_class, style_class in directions:

        print(
            f"\nGenerating: "
            f"shape={shape_class}, "
            f"style={style_class}")

        shape_indices = class_indices[shape_class]
        style_indices = class_indices[style_class]

        # Deterministic selection
        rng = np.random.default_rng(
            SEED
            + CLASS_NAMES.index(shape_class)
            * 100
            + CLASS_NAMES.index(style_class))

        shape_selected = rng.choice(
            shape_indices,
            size=conflicts_per_direction,
            replace=False)

        style_selected = rng.choice(
            style_indices,
            size=conflicts_per_direction,
            replace=False)

        direction_count = 0

        for content_index, style_index in zip(
            shape_selected,
            style_selected):

            content_image, content_label = (
                dataset[int(content_index)])

            style_image, style_label = (
                dataset[int(style_index)])

            # AdaIN
            stylized_image = adain.transfer(
                content_image,
                style_image,
                alpha=STYLE_ALPHA)

            # Visual rejection
            accepted, reason = (
                visually_valid_conflict(
                    content_image,
                    stylized_image))

            if not accepted:

                rejected_count += 1

                records.append({
                    "content_class": shape_class,
                    "style_class": style_class,
                    "content_index": int(
                        content_index
                    ),
                    "style_index": int(
                        style_index
                    ),
                    "status": "rejected",
                    "rejection_reason": reason,
                    "image_path": "",
                })

                continue

            # Save accepted image
            filename = (
                f"shape_{shape_class}"
                f"_style_{style_class}"
                f"_{direction_count:04d}.png")

            output_path = (
                OUTPUT_DIR / filename)

            stylized_image.save(
                output_path)
                
            # Record metadata
            records.append({
                "content_class": shape_class,
                "style_class": style_class,
                "content_index": int(
                    content_index
                ),
                "style_index": int(
                    style_index
                ),
                "status": "accepted",
                "rejection_reason": "",
                "image_path": str(
                    output_path
                ),
            })

            accepted_count += 1
            direction_count += 1

            print(
                f"Accepted: "
                f"{accepted_count}"
            )

    # ========================================================
    # Save metadata
    # ========================================================

    metadata = pd.DataFrame(records)

    metadata_file = (
        PROJECT_ROOT
        / "results"
        / "cue_conflict_metadata.csv"
    )

    metadata.to_csv(
        metadata_file,
        index=False
    )

    # ========================================================
    # Summary
    # ========================================================

    accepted = (
        metadata["status"] == "accepted"
    ).sum()

    rejected = (
        metadata["status"] == "rejected"
    ).sum()

    print("\n" + "=" * 70)
    print("CUE-CONFLICT GENERATION COMPLETE")
    print("=" * 70)

    print(
        "Accepted:",
        accepted
    )

    print(
        "Rejected:",
        rejected
    )

    print(
        "Total attempted:",
        len(metadata)
    )

    print(
        "\nMetadata saved to:"
    )

    print(metadata_file)


if __name__ == "__main__":
    generate_conflicts()