"""
Configuration parameters for Brain MRI Classification pipeline.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
FIGURES_DIR = BASE_DIR / "figures"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

# Checkpoint paths
BEST_EFFICIENTNET_PATH = CHECKPOINTS_DIR / "best_efficientnet_b0.keras"
BEST_RESNET_PATH = CHECKPOINTS_DIR / "best_resnet50.weights.h5"
BEST_RESNET_FT_PATH = CHECKPOINTS_DIR / "best_resnet50_fine_tuning.weights.h5"

# Hyperparameters
RANDOM_SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
RESNET_FINE_TUNE_LR = 1e-5
NUM_EPOCHS = 10
NUM_FINE_TUNE_EPOCHS = 10

# Classes
CLASS_NAMES = ["glioma", "healthy", "meningioma", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)
