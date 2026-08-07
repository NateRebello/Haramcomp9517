"""
Grad-CAM helpers for ResNet-18 species classifiers.

Paper: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization," ICCV 2017.
https://doi.org/10.1109/ICCV.2017.74

Implementation uses the `pytorch-grad-cam` package (Jacob Gildenblat et al.)
targeting ResNet-18's last convolutional block (`layer4[-1]`).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .config import IMAGENET_MEAN, IMAGENET_STD
from .dataset import denormalize


def resnet18_target_layers(model: nn.Module) -> List[nn.Module]:
    """Last conv block — standard Grad-CAM target for ResNet-18."""
    return [model.layer4[-1]]


def make_gradcam(model: nn.Module) -> GradCAM:
    """Build a GradCAM object for a ResNet-18 classifier."""
    return GradCAM(model=model, target_layers=resnet18_target_layers(model))


@torch.no_grad()
def predict_topk(
    model: nn.Module,
    images: torch.Tensor,
    *,
    k: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (top1_idx [N], topk_idx [N,K]) on the same device as images."""
    model.eval()
    logits = model(images)
    top1 = logits.argmax(dim=1)
    topk = logits.topk(min(k, logits.size(1)), dim=1).indices
    return top1, topk


def tensor_to_uint8_rgb(image_chw: torch.Tensor) -> np.ndarray:
    """
    CHW ImageNet-normalised float tensor → HWC float RGB in [0, 1]
    suitable for `show_cam_on_image`.
    """
    rgb = denormalize(image_chw.detach().cpu()).permute(1, 2, 0).numpy()
    return np.clip(rgb, 0.0, 1.0)


def compute_cam_overlay(
    cam: GradCAM,
    image_bchw: torch.Tensor,
    *,
    target_category: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Grad-CAM for a *single* image batch (N=1).

    Returns
    -------
    grayscale_cam : (H, W) float
    overlay_rgb   : (H, W, 3) uint8 visualisation
    """
    assert image_bchw.ndim == 4 and image_bchw.size(0) == 1
    targets = None
    if target_category is not None:
        targets = [ClassifierOutputTarget(int(target_category))]

    # pytorch-grad-cam expects a torch tensor batch
    grayscale = cam(input_tensor=image_bchw, targets=targets)[0]  # (H, W)
    rgb = tensor_to_uint8_rgb(image_bchw[0])
    overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)
    return grayscale, overlay


def split_correct_incorrect(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return index arrays of correct and incorrect predictions."""
    correct = np.where(y_true == y_pred)[0]
    incorrect = np.where(y_true != y_pred)[0]
    return correct, incorrect


def sample_indices(idxs: Sequence[int], n: int, rng: np.random.Generator) -> List[int]:
    """Sample up to n indices without replacement."""
    idxs = list(idxs)
    if len(idxs) <= n:
        return idxs
    chosen = rng.choice(idxs, size=n, replace=False)
    return [int(i) for i in chosen]
