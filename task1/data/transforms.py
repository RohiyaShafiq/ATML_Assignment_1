import random
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


# convert image to rgb and then resize it 224x224
def to_common_image(image, size=224):

    if not isinstance(image, Image.Image):
        image = TF.to_pil_image(image)

    image = image.convert("RGB")
    image = image.resize(
        (size, size),
        Image.Resampling.BILINEAR)

    return image

# convert image to grayscale
def grayscale(image):

    image = to_common_image(image)

    image = TF.rgb_to_grayscale(
        image,
        num_output_channels=3)

    return image

# rotate the image while preserving the geometry
def hue_rotation(image, degrees=60):
    
    image = to_common_image(image)

    # Convert degree range to [-0.5, 0.5]
    factor = degrees / 360.0

    transformed = TF.adjust_hue(
        image,
        hue_factor=factor
    )

    return transformed

# translate the image left , right, top and bottom by creating the reflection 
def translate_image(image, dx, dy):
    """
    dx > 0  : right
    dx < 0  : left

    dy > 0  : down
    dy < 0  : up
    """

    image = to_common_image(image)

    width, height = image.size

    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)

    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)

    # convert image to numoy
    arr = np.asarray(image)

    padded = np.pad(
        arr,
        (
            (abs(dy), abs(dy)),
            (abs(dx), abs(dx)),
            (0, 0)
        ),
        mode="reflect")

    center_y = abs(dy) - dy
    center_x = abs(dx) - dx

    cropped = padded[
        center_y:center_y + height,
        center_x:center_x + width]

    return Image.fromarray(cropped.astype(np.uint8))


# divides the image of 224x224 into 4x4 and randomly permute them 
def patch_shuffle(image, seed=6304, patch_size=4):

    image = to_common_image(image)

    arr = np.asarray(image)

    height, width, channels = arr.shape

    assert height % patch_size == 0
    assert width % patch_size == 0

    rows = height // patch_size
    cols = width // patch_size

    patches = []

    for r in range(rows):
        for c in range(cols):

            patch = arr[
                r * patch_size:(r + 1) * patch_size,
                c * patch_size:(c + 1) * patch_size
            ]

            patches.append(patch)

    rng = np.random.default_rng(seed)

    permutation = rng.permutation(len(patches))

    # Guarantee a non-identity permutation
    while np.array_equal(
        permutation,
        np.arange(len(patches))
    ):
        permutation = rng.permutation(len(patches))

    shuffled = np.zeros_like(arr)

    for new_position, old_position in enumerate(permutation):

        r = new_position // cols
        c = new_position % cols

        shuffled[
            r * patch_size:(r + 1) * patch_size,
            c * patch_size:(c + 1) * patch_size
        ] = patches[old_position]

    return Image.fromarray(shuffled)

# Model normalization
IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225]

# apply imagenet normalization after 224x224 image is construected
def imagenet_normalize(image):

    image = to_common_image(image)

    tensor = TF.to_tensor(image)

    tensor = TF.normalize(
        tensor,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )

    return tensor