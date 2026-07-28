"""
Shared configuration for the COMP9517 deep-learning pipeline.

Keep paths, seed, and default hyperparameters here so notebooks stay thin
and Nate / Abdoali do not drift onto different settings.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Pipeline root: .../Group_project/dl_pipeline
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Native-resolution iNat subset (not the resized copy)
DATA_ROOT = Path(r"C:\Users\Abdoali\Comp9517\Group_project\inat_subset\subset")

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

# ---------------------------------------------------------------------------
# Image / loader defaults (tuned for RTX 2050 4GB)
# ---------------------------------------------------------------------------
IMG_SIZE = 224  # standard ImageNet input; keeps ResNet-18 comfortable in 4GB
NUM_CLASSES = 500

# Training defaults — notebooks may override. Batch 16 + AMP is the safe
# starting point on 4GB; drop to 8 if you OOM.
BATCH_SIZE = 16
NUM_WORKERS = 0  # Windows + Jupyter: 0 avoids multiprocessing hangs
PIN_MEMORY = True  # faster host->GPU copies when CUDA is available

# ImageNet channel stats (used for pretrained models; from-scratch can share
# the same normalisation so both runs see identical pixel distributions).
# Source: torchvision ImageNet presets / PyTorch docs.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN is slower but removes a source of run-to-run noise.
    # Keep False while iterating; flip to True for final report runs if needed.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_output_dirs() -> None:
    """Create checkpoint / results folders if missing."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
