"""
src/engine/evaluator.py
=======================
Accuracy, loss, and hardware-efficiency evaluation for APQ-Lite models.

Role in the pipeline
--------------------
The ``Evaluator`` is used at two points:

1. **Baseline benchmarking** — evaluate the full-precision (fp32) model to
   establish the accuracy reference before quantization.

2. **Post-QAT evaluation** — measure the accuracy/loss of the quantized model
   (real-valued in the VS Code environment, or bit-accurate in the MASE
   environment after ``convert`` / ``quantize_transform_pass`` is applied).

Metrics tracked
    - Top-1 accuracy
    - Top-5 accuracy (for ImageNet-scale datasets)
    - Cross-entropy loss
    - Inference latency (wall-clock ms/sample)
    - Optionally: BOPs (bit-operations) via static layer analysis

Results are returned as a structured dict and can be optionally serialised
to JSON for downstream notebook visualisation (``notebooks/Results_Visualization.ipynb``).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate a PyTorch model on accuracy, loss, and throughput.

    Parameters
    ----------
    model:
        Model to evaluate (fp32 or quantized).
    loss_fn:
        Task loss (typically ``nn.CrossEntropyLoss()``).
    device:
        Torch device string.
    topk:
        Tuple of k values for top-k accuracy, e.g. ``(1, 5)``.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        device: str = "cuda",
        topk: Tuple[int, ...] = (1, 5),
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.topk = topk

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataloader: DataLoader,
        label: str = "eval",
    ) -> Dict[str, float]:
        """Run a full evaluation pass over ``dataloader``.

        Parameters
        ----------
        dataloader:
            Test / validation data loader.
        label:
            Descriptive tag included in log output.

        Returns
        -------
        Dict[str, float]
            Keys: ``"loss"``, ``"top1"``, ``"top5"`` (if applicable),
            ``"throughput_samples_per_sec"``.
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        topk_correct: Dict[int, int] = {k: 0 for k in self.topk}

        t_start = time.perf_counter()

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                batch_n = inputs.size(0)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                total_loss += loss.item() * batch_n
                total_samples += batch_n

                for k, n_correct in self._topk_correct(outputs, targets).items():
                    topk_correct[k] += n_correct

        elapsed = time.perf_counter() - t_start

        results: Dict[str, float] = {
            "loss": total_loss / total_samples,
            "throughput_samples_per_sec": total_samples / elapsed,
        }
        for k in self.topk:
            results[f"top{k}"] = topk_correct[k] / total_samples * 100.0

        self._log_results(label, results)
        return results

    def benchmark_latency(
        self,
        dataloader: DataLoader,
        warmup_batches: int = 5,
        measure_batches: int = 50,
    ) -> Dict[str, float]:
        """Measure per-sample inference latency (ms).

        Parameters
        ----------
        dataloader:
            Data source (only the first ``warmup_batches + measure_batches``
            batches are used).
        warmup_batches:
            Batches to discard before timing starts (GPU warm-up).
        measure_batches:
            Batches over which latency is averaged.

        Returns
        -------
        Dict[str, float]
            ``"latency_ms_per_sample"``, ``"latency_ms_per_batch"``.
        """
        self.model.eval()
        loader_iter = iter(dataloader)

        # Warm-up
        with torch.no_grad():
            for _ in range(warmup_batches):
                try:
                    inputs, _ = next(loader_iter)
                    self.model(inputs.to(self.device))
                except StopIteration:
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_samples = 0
        t_start = time.perf_counter()

        with torch.no_grad():
            for step in range(measure_batches):
                try:
                    inputs, _ = next(loader_iter)
                except StopIteration:
                    break
                inputs = inputs.to(self.device)
                self.model(inputs)
                total_samples += inputs.size(0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "latency_ms_per_sample": elapsed_ms / max(total_samples, 1),
            "latency_ms_per_batch": elapsed_ms / max(step + 1, 1),
        }

    def save_results(
        self,
        results: Dict[str, float],
        output_path: str | Path,
        label: str = "eval",
    ) -> None:
        """Serialise evaluation results to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {"label": label, "metrics": results}
        with open(output_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Evaluation results saved to %s", output_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _topk_correct(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[int, int]:
        """Return the number of correct predictions for each k in self.topk."""
        max_k = max(self.topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(min(max_k, outputs.size(1)), dim=1, largest=True, sorted=True)
        pred = pred.t()                        # (k, batch)
        correct = pred.eq(targets.unsqueeze(0).expand_as(pred))

        results: Dict[int, int] = {}
        for k in self.topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            results[k] = int(correct_k.item())
        return results

    @staticmethod
    def _log_results(label: str, results: Dict[str, float]) -> None:
        parts = [f"[{label}]"]
        if "loss" in results:
            parts.append(f"loss={results['loss']:.4f}")
        for key in sorted(results):
            if key.startswith("top"):
                parts.append(f"{key}={results[key]:.2f}%")
        if "throughput_samples_per_sec" in results:
            parts.append(
                f"throughput={results['throughput_samples_per_sec']:.1f} samp/s"
            )
        logger.info("  ".join(parts))
