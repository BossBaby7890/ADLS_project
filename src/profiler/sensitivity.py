"""
src/profiler/sensitivity.py
===========================
Stage 1 of the APQ-Lite (HAWQ-lite) pipeline: **Gradient Sensitivity Profiling**.

Role in the pipeline
--------------------
HAWQ (Hessian-Aware Quantization Weighting) uses the top eigenvalue of the
per-layer Hessian as a sensitivity metric — expensive for large networks.
APQ-Lite approximates this with the **mean squared L2 norm of the gradients**
over a small calibration set.  Empirically, high-gradient-norm layers are
also high-Hessian-eigenvalue layers, making this a cheap 1st-order proxy.

Concretely, for each parametric layer l:

    sensitivity(l) = (1 / N) * sum_{i=1}^{N} || grad_W_l(x_i) ||_2^2

where N is the number of calibration batches and grad_W_l is the gradient
of the task loss w.r.t. layer l's weights.

The resulting scores are (optionally) min-max normalised to [0, 1] and
serialised to JSON for consumption by `src/allocator/bit_mapper.py`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class GradientSensitivityProfiler:
    """Compute per-layer squared gradient norms as a Hessian proxy.

    Parameters
    ----------
    model:
        The (full-precision) PyTorch model to profile.
    loss_fn:
        Task loss callable, e.g. ``torch.nn.CrossEntropyLoss()``.
    device:
        Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
    normalize:
        If ``True``, min-max normalise scores to [0, 1] before returning.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        device: str = "cuda",
        normalize: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.normalize = normalize

        # Accumulator: layer_name -> cumulative squared gradient norm
        self._scores: Dict[str, float] = {}
        self._batch_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(
        self,
        dataloader: DataLoader,
        num_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run the profiling pass over `num_batches` mini-batches.

        Parameters
        ----------
        dataloader:
            Calibration dataloader (typically a small subset of training data).
        num_batches:
            Stop after this many batches.  ``None`` uses the full loader.

        Returns
        -------
        Dict[str, float]
            Mapping from layer name to (normalised) sensitivity score.
        """
        self.model.eval()
        self._reset()

        limit = num_batches or len(dataloader)
        logger.info("Profiling sensitivity over %d batches ...", limit)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= limit:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()

            self._accumulate_gradients()
            self._batch_count += 1

            if (batch_idx + 1) % 10 == 0:
                logger.debug("  Processed %d / %d batches", batch_idx + 1, limit)

        scores = self._finalise()
        logger.info("Profiling complete. %d layers scored.", len(scores))
        return scores

    def save(self, output_path: str | Path) -> None:
        """Serialise the last computed scores to a JSON file."""
        if not self._scores:
            raise RuntimeError("Call .profile() before .save().")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(self._scores, fh, indent=2)
        logger.info("Sensitivity scores saved to %s", output_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._scores = {}
        self._batch_count = 0

    def _accumulate_gradients(self) -> None:
        """Add squared gradient norms for the current backward pass."""
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            # Squared L2 norm of the gradient tensor
            sq_norm = param.grad.detach().pow(2).sum().item()
            self._scores[name] = self._scores.get(name, 0.0) + sq_norm

    def _finalise(self) -> Dict[str, float]:
        """Average over batches and optionally normalise."""
        if self._batch_count == 0:
            return {}

        # Average across batches
        averaged: Dict[str, float] = {
            k: v / self._batch_count for k, v in self._scores.items()
        }

        if self.normalize:
            values = list(averaged.values())
            min_v, max_v = min(values), max(values)
            spread = max_v - min_v if max_v != min_v else 1.0
            averaged = {k: (v - min_v) / spread for k, v in averaged.items()}

        self._scores = averaged
        return averaged

    # ------------------------------------------------------------------
    # Convenience: aggregate param-level scores to layer-level
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_to_layers(
        param_scores: Dict[str, float],
        model: nn.Module,
    ) -> Dict[str, float]:
        """Collapse per-parameter scores to per-named-module scores.

        For a layer with multiple parameters (weight + bias), the score is
        taken as the maximum parameter score — conservative & safe.

        Parameters
        ----------
        param_scores:
            Output of :meth:`profile` (keyed by ``named_parameters`` names).
        model:
            The same model used during profiling.

        Returns
        -------
        Dict[str, float]
            Mapping from module name (e.g. ``"layer1.0.conv1"``) to score.
        """
        layer_scores: Dict[str, float] = {}
        for mod_name, _ in model.named_modules():
            matching = {
                k: v
                for k, v in param_scores.items()
                if k.startswith(mod_name + ".") or k == mod_name
            }
            if matching:
                layer_scores[mod_name] = max(matching.values())
        return layer_scores
