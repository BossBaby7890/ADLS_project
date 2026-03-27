"""
src/pruning/kd_recovery.py
==========================
Stage 2.2 — **Knowledge Distillation Recovery** after structured pruning.

Reference
---------
Hinton et al., "Distilling the Knowledge in a Neural Network",
NeurIPS 2014 Workshop.  https://arxiv.org/abs/1503.02531

Why KD instead of standard fine-tuning
---------------------------------------
After aggressively removing channels, the pruned student model can lose
several accuracy points.  Standard fine-tuning against hard one-hot labels
provides a weak gradient signal: every sample contributes a gradient only
from the correct-class logit.

Knowledge distillation matches the student's output distribution to the
teacher's *soft* logits.  The teacher's probability vector encodes
inter-class similarity (e.g. "70% cat, 20% dog") which provides richer
gradients and accelerates recovery, especially at high sparsity.

The FP32 model used for profiling in Stage 1 is already in memory and
serves as the teacher for free — no additional training is needed.

Loss formulation
----------------
    L_KD = KL( σ(z_teacher / T) || σ(z_student / T) ) · T²

where T is the temperature (higher T = softer distributions, more
inter-class information transferred).  We optionally blend with hard
cross-entropy:

    L = α · L_KD + (1 − α) · L_CE

The T² scaling factor restores gradient magnitude after softmax division.

Usage
-----
>>> from src.pruning.kd_recovery import KDRecovery
>>> kd = KDRecovery(teacher_model, device="cuda", epochs=5, temperature=4.0)
>>> kd.recover(student_model, dataloader)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class KDRecovery:
    """Fine-tune a pruned student model using a frozen FP32 teacher.

    Parameters
    ----------
    teacher:
        The original full-precision model (frozen during recovery).
    device:
        Torch device string.
    epochs:
        Number of recovery fine-tune epochs.
    temperature:
        Distillation temperature T.  Higher values soften distributions.
        Hinton et al. recommend T ∈ [2, 10].
    alpha:
        Weight for the KD loss vs. hard CE loss.
        ``1.0`` = pure KD,  ``0.0`` = standard CE fine-tune.
    learning_rate:
        Learning rate for the student optimizer (SGD with momentum).
    momentum:
        SGD momentum.
    weight_decay:
        L2 regularisation on student weights.
    """

    def __init__(
        self,
        teacher: nn.Module,
        device: str = "cuda",
        epochs: int = 5,
        temperature: float = 4.0,
        alpha: float = 0.9,
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ) -> None:
        self.device = device
        self.epochs = epochs
        self.temperature = temperature
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Freeze teacher and move to device
        self.teacher = teacher.to(device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recover(
        self,
        student: nn.Module,
        dataloader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Fine-tune ``student`` against the frozen teacher.

        Parameters
        ----------
        student:
            The pruned model to recover.  Updated in-place.
        dataloader:
            Training dataloader (or calibration subset).
        val_loader:
            Optional validation loader for logging accuracy per epoch.

        Returns
        -------
        nn.Module
            The recovered student model (same object, updated in-place).
        """
        student = student.to(self.device)
        student.train()

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, student.parameters()),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        ce_loss_fn = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            student.train()
            total_loss = 0.0
            n_batches = 0

            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()

                # Student forward
                s_logits = student(inputs)

                # Teacher forward (no grad)
                with torch.no_grad():
                    t_logits = self.teacher(inputs)

                # KD loss: KL divergence between soft distributions
                kd_loss = self._kd_loss(s_logits, t_logits)

                # Hard CE loss
                ce_loss = ce_loss_fn(s_logits, targets)

                # Blended loss
                loss = self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)

            if val_loader is not None:
                val_acc = self._eval_accuracy(student, val_loader)
                logger.info(
                    "KD recovery epoch %d/%d — loss=%.4f  val_acc=%.2f%%",
                    epoch + 1, self.epochs, avg_loss, val_acc * 100,
                )
            else:
                logger.info(
                    "KD recovery epoch %d/%d — loss=%.4f",
                    epoch + 1, self.epochs, avg_loss,
                )

        student.eval()
        return student

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence loss between softened student and teacher logits.

        L_KD = KL(teacher_soft || student_soft) · T²

        The T² factor restores gradient magnitude that is lost by dividing
        logits by T before softmax.
        """
        T = self.temperature
        s_soft = F.log_softmax(student_logits / T, dim=1)
        t_soft = F.softmax(teacher_logits / T, dim=1)
        # kl_div expects log-probabilities for input, probabilities for target
        kd = F.kl_div(s_soft, t_soft, reduction="batchmean") * (T ** 2)
        return kd

    @torch.no_grad()
    def _eval_accuracy(self, model: nn.Module, loader: DataLoader) -> float:
        """Compute top-1 accuracy on ``loader``."""
        model.eval()
        correct = 0
        total = 0
        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        return correct / max(total, 1)
