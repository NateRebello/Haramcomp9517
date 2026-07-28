"""
Training / evaluation helpers with AMP, timing, and checkpointing.

Designed for the RTX 2050 (4GB): mixed precision is on by default.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction correct (0–1) for a single batch."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    *,
    use_amp: bool = True,
    max_batches: Optional[int] = None,
    log_every: int = 50,
) -> Dict[str, float]:
    """
    Run one training epoch (or a short smoke subset via ``max_batches``).

    Returns mean loss, mean top-1 accuracy, and wall-clock seconds.
    """
    model.train()
    if use_amp and scaler is None and device.type == "cuda":
        scaler = GradScaler("cuda")

    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0
    t0 = time.perf_counter()

    pbar = tqdm(loader, desc="train", leave=False)
    for step, (images, targets) in enumerate(pbar):
        if max_batches is not None and step >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with autocast("cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        batch_acc = accuracy_top1(logits.detach(), targets)
        running_loss += loss.item()
        running_acc += batch_acc
        n_batches += 1

        if step % log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{batch_acc:.3f}")

    elapsed = time.perf_counter() - t0
    return {
        "loss": running_loss / max(n_batches, 1),
        "acc": running_acc / max(n_batches, 1),
        "seconds": elapsed,
        "batches": float(n_batches),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    use_amp: bool = True,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Validation / test pass: mean loss + top-1 accuracy + seconds."""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0
    t0 = time.perf_counter()

    for step, (images, targets) in enumerate(tqdm(loader, desc="eval", leave=False)):
        if max_batches is not None and step >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with autocast("cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
        else:
            logits = model(images)
            loss = criterion(logits, targets)

        running_loss += loss.item()
        running_acc += accuracy_top1(logits, targets)
        n_batches += 1

    elapsed = time.perf_counter() - t0
    return {
        "loss": running_loss / max(n_batches, 1),
        "acc": running_acc / max(n_batches, 1),
        "seconds": elapsed,
        "batches": float(n_batches),
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    epoch: int,
    history: Dict[str, List[float]],
    config: Dict[str, Any],
    class_to_idx: Dict[str, int],
    best_val_acc: float,
) -> None:
    """Save model + optimizer (+ AMP scaler) so training can resume after a crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "history": history,
        "config": config,
        "class_to_idx": class_to_idx,
        "best_val_acc": best_val_acc,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Restore weights (and optionally optimizer / scaler). Returns the full payload."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt


def save_history_json(path: Path, history: Dict[str, List[float]], config: Dict[str, Any]) -> None:
    """Write loss/acc curves + hyperparams as JSON for the report / notebook 05."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"config": config, "history": history}, f, indent=2)


def estimate_vram_mb(device: torch.device) -> Tuple[float, float]:
    """Return (allocated_MB, reserved_MB) on CUDA; (0, 0) on CPU."""
    if device.type != "cuda":
        return 0.0, 0.0
    alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    return alloc, reserved
