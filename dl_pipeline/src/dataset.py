"""
Dataset and DataLoader helpers for the iNaturalist-2021 subset.

Folder layout expected (ImageFolder-compatible)::

    DATA_ROOT/
      train/<category_id>/*.jpg   # 40 images / class
      val/<category_id>/*.jpg     # 10 images / class
      test/<category_id>/*.jpg    # 10 images / class  (official iNat val)

Category folder names are iNat category IDs (e.g. ``1022``). Class indices
are assigned by sorting those names alphabetically — the same rule
``torchvision.datasets.ImageFolder`` uses — and we assert train/val/test
share an identical ``class_to_idx`` so labels never silently disagree.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import (
    BATCH_SIZE,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_WORKERS,
    PIN_MEMORY,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
)


def get_transforms(
    split: str,
    img_size: int = IMG_SIZE,
    *,
    augment_train: bool = True,
) -> transforms.Compose:
    """
    Build preprocessing transforms.

    Parameters
    ----------
    split :
        ``\"train\"``, ``\"val\"``, or ``\"test\"``.
    img_size :
        Square resize target (default 224).
    augment_train :
        If True, apply light augmentation on the train split only.
        Keep this modest at first; heavy aug is an ablation later.

    Notes
    -----
    Resize strategy: ``Resize(img_size)`` then ``CenterCrop(img_size)`` for
    eval; train uses ``RandomResizedCrop`` when augmenting. This matches
    common ImageNet transfer-learning practice (He et al., ResNet; torchvision
    training recipes).
    """
    split = split.lower()
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train" and augment_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    # val / test / train-without-aug: deterministic preprocessing
    return transforms.Compose(
        [
            transforms.Resize(img_size + 32),  # short-side-ish buffer before crop
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def _assert_same_classes(
    reference: datasets.ImageFolder,
    other: datasets.ImageFolder,
    other_name: str,
) -> None:
    if reference.class_to_idx != other.class_to_idx:
        raise RuntimeError(
            f"{other_name} class_to_idx differs from train. "
            "Train/val/test must contain the same category folders."
        )


def build_datasets(
    train_dir: Path = TRAIN_DIR,
    val_dir: Path = VAL_DIR,
    test_dir: Path = TEST_DIR,
    img_size: int = IMG_SIZE,
    *,
    augment_train: bool = True,
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    """Create train / val / test ImageFolder datasets with aligned labels."""
    train_ds = datasets.ImageFolder(
        root=str(train_dir),
        transform=get_transforms("train", img_size, augment_train=augment_train),
    )
    val_ds = datasets.ImageFolder(
        root=str(val_dir),
        transform=get_transforms("val", img_size),
    )
    test_ds = datasets.ImageFolder(
        root=str(test_dir),
        transform=get_transforms("test", img_size),
    )

    _assert_same_classes(train_ds, val_ds, "val")
    _assert_same_classes(train_ds, test_ds, "test")
    return train_ds, val_ds, test_ds


def build_dataloaders(
    train_ds: datasets.ImageFolder,
    val_ds: datasets.ImageFolder,
    test_ds: datasets.ImageFolder,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap datasets in DataLoaders.

    ``pin_memory`` defaults to True when CUDA is available (see config),
    which speeds up transfers on the RTX 2050 without using extra GPU VRAM
    for the model itself.
    """
    if pin_memory is None:
        pin_memory = PIN_MEMORY and torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def summarize_dataset(ds: datasets.ImageFolder, name: str) -> Dict:
    """Return a small dict of counts useful for logging / the report."""
    # targets is a list of int labels, one per image
    per_class = Counter(ds.targets)
    counts = list(per_class.values())
    return {
        "split": name,
        "num_images": len(ds),
        "num_classes": len(ds.classes),
        "images_per_class_min": min(counts) if counts else 0,
        "images_per_class_max": max(counts) if counts else 0,
        "class_id_examples": ds.classes[:5],
    }


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD,
) -> torch.Tensor:
    """
    Undo ImageNet normalisation for matplotlib display.

    Expects a CHW or NCHW float tensor. Returns a cloned tensor in [0, 1].
    """
    t = tensor.clone().detach()
    if t.ndim == 3:
        for c, m, s in zip(range(t.shape[0]), mean, std):
            t[c] = t[c] * s + m
    elif t.ndim == 4:
        for c, m, s in zip(range(t.shape[1]), mean, std):
            t[:, c] = t[:, c] * s + m
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(t.shape)}")
    return t.clamp(0.0, 1.0)


def idx_to_category_id(ds: datasets.ImageFolder) -> Dict[int, str]:
    """Map integer class index -> original iNat category-id folder name."""
    return {idx: name for name, idx in ds.class_to_idx.items()}
