"""
src/profiler/sensitivity.py
===========================
Stage 1 — **Joint Sensitivity Profiling** (extended).

Three metrics are computed in a single calibration pass and written to disk:

Metric 1 — Gradient norm  (existing, unchanged)
------------------------------------------------
    sensitivity(l) = (1/N) Σ_i  ||∇_W_l L(x_i)||²

    1st-order Hessian proxy (HAWQ-lite).  Used as a feature in the accuracy
    proxy MLP and as the signal for the greedy allocator fallback.

Metric 2 — Hutchinson Hessian trace  (new)
------------------------------------------
    tr(H_l) ≈ (1/M) Σ_j  z_j^T · (∇²_W_l L) · z_j
             = (1/M) Σ_j  (∇_W_l [∇_W_l L · z_j]) · z_j

    where z_j ~ Rademacher{±1}.  Computed via two backward passes per
    probe vector using the Pearlmutter trick (Hessian-vector product).
    Captures *curvature*, not just gradient magnitude — a layer can have
    a large gradient norm but flat curvature, meaning it tolerates more
    compression.  Used alongside metric 1 as a richer feature vector for
    the accuracy proxy.

    Reference: Yao et al., "PyHessian", ICDM 2020.

Metric 3 — Taylor channel importance scores  (new)
--------------------------------------------------
    I(c) = |E_x[ a_c(x) · (∂L/∂a_c)(x) ]|

    where a_c is the output activation of channel c and ∂L/∂a_c is its
    gradient.  This is the 1st-order Taylor expansion of the loss change
    from removing channel c (Molchanov et al., ICLR 2017).

    Unlike metrics 1 and 2, this operates at *channel* granularity inside
    each conv layer, not at layer granularity.  It is consumed exclusively
    by `src/pruning/taylor_pruner.py` to decide which channels to prune.

Output schema
-------------
Two JSON files are written:

``outputs/sensitivity_scores.json``  — parameter-level gradient norms
``outputs/layer_sensitivity.json``   — module-level aggregated dict::

    {
        "layer1.0.conv1": {
            "grad_norm":       0.82,   # normalised gradient norm
            "hessian_trace":   0.61,   # normalised Hutchinson trace
            "taylor_scores":   [0.3, 0.8, 0.1, ...]  # per-channel (conv only)
        },
        ...
    }

The ``grad_norm`` and ``hessian_trace`` fields are both normalised to [0, 1]
across all profiled layers.  ``taylor_scores`` are normalised per-layer so
the channel with highest importance = 1.0.

Backward compatibility
----------------------
``GradientSensitivityProfiler`` preserves its original public interface
(``profile()``, ``save()``, ``aggregate_to_layers()``) so that
``scripts/run_profiling.py`` requires no changes.  The richer output is
written to ``layer_sensitivity.json`` by the new ``save_full()`` method.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hutchinson Hessian-vector product helper
# ---------------------------------------------------------------------------

def _hvp(loss: torch.Tensor, params: List[torch.Tensor], v: List[torch.Tensor]) -> List[torch.Tensor]:
    """Compute the Hessian-vector product H·v via two backward passes.

    Uses the Pearlmutter trick:
        H·v = ∇_θ [ (∇_θ L)^T · v ]

    Parameters
    ----------
    loss:
        Scalar loss (must be from a fresh forward pass with create_graph=True).
    params:
        List of parameter tensors (must have requires_grad=True).
    v:
        List of probe vectors, same shapes as params.

    Returns
    -------
    List of Hessian-vector product tensors, one per parameter.
    """
    # First-order gradients with graph retained for second differentiation
    grads = torch.autograd.grad(
        loss, params, create_graph=True, retain_graph=True, allow_unused=True
    )
    # Dot product (∇L)^T · v  — scalar
    gv = sum(
        (g * vi).sum()
        for g, vi in zip(grads, v)
        if g is not None
    )
    # Second backward: ∇_θ [ (∇L)^T · v ] = H·v
    hvp_list = torch.autograd.grad(
        gv, params, retain_graph=False, allow_unused=True
    )
    return [h if h is not None else torch.zeros_like(p) for h, p in zip(hvp_list, params)]


# ---------------------------------------------------------------------------
# Main profiler class
# ---------------------------------------------------------------------------

class GradientSensitivityProfiler:
    """Joint profiler: gradient norm + Hutchinson trace + Taylor scores.

    Parameters
    ----------
    model:
        Full-precision PyTorch model to profile.
    loss_fn:
        Task loss callable, e.g. ``nn.CrossEntropyLoss()``.
    device:
        Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
    normalize:
        If ``True``, min-max normalise gradient-norm and Hessian-trace
        scores to [0, 1] before returning.
    hutchinson_probes:
        Number of Rademacher probe vectors used to estimate the Hessian
        trace per batch.  More probes → lower variance, higher cost.
        Paper recommends 1-5 for a lightweight estimate.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        device: str = "cuda",
        normalize: bool = True,
        hutchinson_probes: int = 3,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.normalize = normalize
        self.hutchinson_probes = hutchinson_probes

        # --- Accumulators (reset at each profile() call) ---
        # Metric 1: parameter-level gradient norms
        self._grad_scores: Dict[str, float] = {}
        # Metric 2: module-level Hutchinson trace estimates
        self._hess_scores: Dict[str, float] = {}
        # Metric 3: channel-level Taylor importance (module -> list[float])
        self._taylor_scores: Dict[str, List[float]] = {}
        # Activation hooks storage
        self._activations: Dict[str, torch.Tensor] = {}
        self._act_grads: Dict[str, torch.Tensor] = {}
        self._batch_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(
        self,
        dataloader: DataLoader,
        num_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run the joint profiling pass.

        Preserves the original ``GradientSensitivityProfiler.profile()``
        return signature (parameter-level gradient norms) for backward
        compatibility with ``run_profiling.py``.

        Parameters
        ----------
        dataloader:
            Calibration dataloader.
        num_batches:
            Stop after this many batches.  ``None`` uses the full loader.

        Returns
        -------
        Dict[str, float]
            Parameter-level (normalised) gradient-norm scores.
            Call ``get_full_layer_scores()`` for all three metrics.
        """
        self._reset()
        hooks = self._register_taylor_hooks()

        limit = num_batches or len(dataloader)
        logger.info("Joint profiling over %d batches ...", limit)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= limit:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # ---- Forward + backward for gradient norm & Taylor scores ----
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()

            self._accumulate_grad_norms()
            self._accumulate_taylor_scores()

            # ---- Hutchinson trace (separate passes, one per probe) --------
            self._accumulate_hutchinson(inputs, targets)

            self._batch_count += 1

            if (batch_idx + 1) % 10 == 0:
                logger.debug("  Processed %d / %d batches", batch_idx + 1, limit)

        for h in hooks:
            h.remove()

        scores = self._finalise_grad_norms()
        logger.info("Joint profiling complete. %d parameters scored.", len(scores))
        return scores

    def get_full_layer_scores(self, model: nn.Module) -> Dict[str, dict]:
        """Return the full per-layer score dict with all three metrics.

        Must be called *after* ``profile()``.

        Parameters
        ----------
        model:
            The same model used during profiling (needed for aggregation).

        Returns
        -------
        Dict[str, dict]
            Keys are module names.  Each value is::

                {
                    "grad_norm":     float,   # normalised ∈ [0, 1]
                    "hessian_trace": float,   # normalised ∈ [0, 1]
                    "taylor_scores": list[float]  # per-channel, normalised
                }
        """
        if self._batch_count == 0:
            raise RuntimeError("Call profile() before get_full_layer_scores().")

        # Aggregate param-level grad norms to module level
        layer_grad = self.aggregate_to_layers(self._grad_scores, model)

        # Normalise Hutchinson traces
        layer_hess = dict(self._hess_scores)
        if self.normalize and layer_hess:
            layer_hess = _minmax_normalise(layer_hess)

        # Normalise grad norms at layer level
        if self.normalize and layer_grad:
            layer_grad = _minmax_normalise(layer_grad)

        # Combine into unified dict
        all_modules = set(layer_grad) | set(layer_hess) | set(self._taylor_scores)
        result: Dict[str, dict] = {}
        for name in all_modules:
            result[name] = {
                "grad_norm":     layer_grad.get(name, 0.0),
                "hessian_trace": layer_hess.get(name, 0.0),
                "taylor_scores": self._taylor_scores.get(name, []),
            }
        return result

    def save(self, output_path: str | Path) -> None:
        """Serialise parameter-level gradient norm scores (backward compat)."""
        if not self._grad_scores:
            raise RuntimeError("Call profile() before save().")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(self._grad_scores, fh, indent=2)
        logger.info("Gradient norm scores saved to %s", output_path)

    def save_full(self, output_path: str | Path, model: nn.Module) -> None:
        """Serialise all three metrics to a single JSON file.

        Parameters
        ----------
        output_path:
            Destination path, e.g. ``outputs/layer_sensitivity.json``.
        model:
            The same model used during profiling.
        """
        full = self.get_full_layer_scores(model)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(full, fh, indent=2)
        logger.info("Full layer sensitivity scores saved to %s", output_path)

    # ------------------------------------------------------------------
    # Static helper — preserved from original for downstream compatibility
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_to_layers(
        param_scores: Dict[str, float],
        model: nn.Module,
    ) -> Dict[str, float]:
        """Collapse per-parameter scores to per-named-module scores.

        For a layer with multiple parameters (weight + bias), the score is
        the maximum parameter score — conservative and safe.
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

    # ------------------------------------------------------------------
    # Internal: activation + gradient hooks for Taylor scores
    # ------------------------------------------------------------------

    def _register_taylor_hooks(self) -> list:
        """Register forward + backward hooks on all Conv2d layers."""
        hooks = []

        def make_fwd_hook(name):
            def hook(module, inp, out):
                # Store detached activation for this batch
                self._activations[name] = out.detach()
            return hook

        def make_bwd_hook(name):
            def hook(module, grad_input, grad_output):
                # grad_output[0]: gradient w.r.t. the layer's output activations
                if grad_output[0] is not None:
                    self._act_grads[name] = grad_output[0].detach()
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(make_fwd_hook(name)))
                hooks.append(module.register_backward_hook(make_bwd_hook(name)))

        return hooks

    # ------------------------------------------------------------------
    # Internal: per-batch accumulation
    # ------------------------------------------------------------------

    def _accumulate_grad_norms(self) -> None:
        """Add squared gradient norms for the current backward pass."""
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            sq_norm = param.grad.detach().pow(2).sum().item()
            self._grad_scores[name] = self._grad_scores.get(name, 0.0) + sq_norm

    def _accumulate_taylor_scores(self) -> None:
        """Accumulate |activation × gradient| channel scores.

        Taylor importance for channel c:
            I(c) += |mean_{spatial, batch}(a_c · ∂L/∂a_c)|
        """
        for name in list(self._activations.keys()):
            if name not in self._act_grads:
                continue
            act = self._activations[name]   # (B, C, H, W)
            grad = self._act_grads[name]    # (B, C, H, W)

            # Element-wise product, average over batch and spatial dims
            importance = (act * grad).abs().mean(dim=(0, 2, 3))  # (C,)
            importance = importance.cpu().tolist()

            if name not in self._taylor_scores:
                self._taylor_scores[name] = [0.0] * len(importance)
            for c, val in enumerate(importance):
                self._taylor_scores[name][c] += val

        # Clear buffers for next batch
        self._activations.clear()
        self._act_grads.clear()

    def _accumulate_hutchinson(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Estimate Hessian trace per module via Hutchinson's estimator.

        For each Conv2d / Linear module, probes the Hessian of the loss
        w.r.t. the module's weight using M Rademacher random vectors.

        tr(H) ≈ (1/M) Σ_j  z_j^T H z_j
        """
        for name, module in self.model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            if module.weight is None:
                continue

            trace_estimate = 0.0
            w = module.weight

            for _ in range(self.hutchinson_probes):
                # Rademacher probe vector ±1
                z = torch.randint_like(w, low=0, high=2).float() * 2 - 1

                # Fresh forward pass with graph retained for HVP
                self.model.zero_grad()
                out = self.model(inputs)
                loss = self.loss_fn(out, targets)

                # HVP: H·z via two backward passes
                hvp = _hvp(loss, [w], [z])  # list of 1 tensor
                trace_estimate += (hvp[0] * z).sum().item()

            trace_estimate /= self.hutchinson_probes
            self._hess_scores[name] = (
                self._hess_scores.get(name, 0.0) + trace_estimate
            )

    # ------------------------------------------------------------------
    # Internal: finalise
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._grad_scores = {}
        self._hess_scores = {}
        self._taylor_scores = {}
        self._activations = {}
        self._act_grads = {}
        self._batch_count = 0

    def _finalise_grad_norms(self) -> Dict[str, float]:
        """Average gradient norms over batches and optionally normalise."""
        if self._batch_count == 0:
            return {}

        averaged = {k: v / self._batch_count for k, v in self._grad_scores.items()}
        if self.normalize:
            averaged = _minmax_normalise(averaged)
        self._grad_scores = averaged

        # Average Hutchinson traces
        self._hess_scores = {
            k: v / self._batch_count for k, v in self._hess_scores.items()
        }

        # Average and normalise Taylor scores per layer
        for name in self._taylor_scores:
            scores = [v / self._batch_count for v in self._taylor_scores[name]]
            # Per-layer normalisation: highest channel = 1.0
            max_s = max(scores) if scores else 1.0
            if max_s > 0:
                scores = [s / max_s for s in scores]
            self._taylor_scores[name] = scores

        return averaged


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _minmax_normalise(d: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise a flat dict of floats to [0, 1]."""
    values = list(d.values())
    min_v, max_v = min(values), max(values)
    spread = max_v - min_v if max_v != min_v else 1.0
    return {k: (v - min_v) / spread for k, v in d.items()}
