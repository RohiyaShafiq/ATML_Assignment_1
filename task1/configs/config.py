from pathlib import Path

# creates same random numbers when we run the code
SEED = 6304

# Dataset
DATASET_NAME = "STL10"

IMAGE_SIZE = 224
NUM_CLASSES = 10

VAL_RATIO = 0.20
TEST_SUBSET_SIZE = 500

# hyperparameters
BATCH_SIZE = 64

MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# CLIP model 
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"

CLIP_PROMPT = "a photo of a {}."


# Paths for results
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SUBSET_FILE = RESULTS_DIR / "subset_ids.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)