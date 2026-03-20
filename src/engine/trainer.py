"""
src/engine/trainer.py
=====================
Training engine for both standard (fp32) and Quantization-Aware Training (QAT).

Role in the pipeline
--------------------
The ``Trainer`` class is the **execution backbone** of APQ-Lite.  It is used
at two points in the pipeline:

1. **Full-precision pre-training** — trains the model in fp32 to convergence
   before sensitivity profiling.  The resulting checkpoint is the starting
   point for QAT.

2. **Quantization-Aware Training (QAT)** — fine-tunes the model *with
   fake-quantizers inserted* according to the mixed-precision config generated
   by ``src/compiler/mase_integration.py``.  The fake-quantizers simulate
   quantization noise during the forward pass so that the model learns to be
   robust to quantization error at the assigned bit-widths.

The same ``Trainer`` object handles both modes; callers differentiate by
passing a plain model (fp32) or a model prepared with
``torch.quantization.prepare_qat`` / MASE fake-quant wrappers (QAT).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class AverageMeter:
    """Tracks a running mean of a scalar (loss, accuracy, …)."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer:
    """Unified training loop for fp32 pre-training and QAT fine-tuning.

    Parameters
    ----------
    model:
        PyTorch model (plain or QAT-prepared).
    optimizer:
        Gradient-descent optimizer.
    loss_fn:
        Task loss (e.g. ``nn.CrossEntropyLoss()``).
    device:
        Torch device string.
    scheduler:
        Optional LR scheduler stepped once per epoch.
    grad_clip:
        Maximum gradient norm (0 = disabled).
    checkpoint_dir:
        Directory for saving epoch checkpoints.
    use_amp:
        Enable automatic mixed-precision (fp16) via ``torch.cuda.amp``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        device: str = "cuda",
        scheduler: Optional[_LRScheduler] = None,
        grad_clip: float = 0.0,
        checkpoint_dir: str = "outputs/checkpoints",
        use_amp: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epochs: int,
        start_epoch: int = 0,
        eval_every: int = 1,
        save_best: bool = True,
        extra_callbacks: Optional[List[Callable]] = None,
    ) -> List[Dict[str, float]]:
        """Run the full training loop.

        Parameters
        ----------
        train_loader:
            Training data loader.
        val_loader:
            Validation data loader (``None`` skips validation).
        epochs:
            Total number of epochs to train.
        start_epoch:
            Resume from this epoch (used when reloading checkpoints).
        eval_every:
            Run validation every N epochs.
        save_best:
            If ``True``, save a checkpoint whenever val loss improves.
        extra_callbacks:
            List of ``callback(epoch, history)`` functions called after each epoch.

        Returns
        -------
        List[Dict[str, float]]
            Per-epoch metrics history.
        """
        best_val_loss = float("inf")

        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            train_metrics = self._train_epoch(train_loader)
            elapsed = time.time() - t0

            row: Dict[str, float] = {"epoch": epoch, **train_metrics, "elapsed": elapsed}

            if val_loader is not None and (epoch + 1) % eval_every == 0:
                val_metrics = self._validate_epoch(val_loader)
                row.update({f"val_{k}": v for k, v in val_metrics.items()})

                if save_best and row.get("val_loss", float("inf")) < best_val_loss:
                    best_val_loss = row["val_loss"]
                    self.save_checkpoint(epoch, tag="best")

            self.history.append(row)

            if self.scheduler is not None:
                self.scheduler.step()

            self._log_epoch(row)

            if extra_callbacks:
                for cb in extra_callbacks:
                    cb(epoch, self.history)

        return self.history

    def save_checkpoint(self, epoch: int, tag: str = "last") -> Path:
        """Persist model + optimizer state to disk."""
        path = self.checkpoint_dir / f"checkpoint_{tag}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            path,
        )
        logger.info("Checkpoint saved → %s", path)
        return path

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> int:
        """Load checkpoint; returns the epoch number."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.history = ckpt.get("history", [])
        epoch = ckpt.get("epoch", 0)
        logger.info("Checkpoint loaded from %s (epoch %d)", path, epoch)
        return epoch

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        loss_meter = AverageMeter("loss")
        correct = total = 0

        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_meter.update(loss.item(), inputs.size(0))
            preds = outputs.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total += inputs.size(0)

        return {"loss": loss_meter.avg, "acc": correct / total}

    def _validate_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        loss_meter = AverageMeter("val_loss")
        correct = total = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss_meter.update(loss.item(), inputs.size(0))
                preds = outputs.argmax(dim=1)
                correct += preds.eq(targets).sum().item()
                total += inputs.size(0)

        return {"loss": loss_meter.avg, "acc": correct / total}

    @staticmethod
    def _log_epoch(row: Dict[str, float]) -> None:
        parts = [f"Epoch {int(row['epoch']):3d}"]
        parts.append(f"loss={row.get('loss', 0):.4f}")
        parts.append(f"acc={row.get('acc', 0)*100:.2f}%")
        if "val_loss" in row:
            parts.append(f"val_loss={row['val_loss']:.4f}")
            parts.append(f"val_acc={row.get('val_acc', 0)*100:.2f}%")
        parts.append(f"[{row.get('elapsed', 0):.1f}s]")
        logger.info("  ".join(parts))
