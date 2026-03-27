"""
src/hardware/cost_model.py
==========================
Hardware-aware cost model for the CMA-ES and greedy policy search engines.

Role in the pipeline
--------------------
The search engine evaluates hundreds of candidate compression policies
(pruning ratios + bit-widths per layer).  Running real inference for each
candidate is prohibitively expensive.  This module provides two fast
estimates that are queried instead:

  1. **BOPs (Bit-Operations)** — a hardware-agnostic arithmetic metric.
  2. **Predicted latency (ms)** — a roofline-based estimate calibrated to a
     target hardware class via a small lookup table.

BOPs formula for a Conv2d layer
--------------------------------
    BOPs = 2 · C_in · C_out · K² · H_out · W_out · b_w · b_a · (1 − sparsity)

where b_w and b_a are weight and activation bit-widths.  The (1 − sparsity)
factor accounts for structured channel pruning: removing a fraction of
output channels reduces the effective C_out proportionally.

For a Linear layer:
    BOPs = 2 · C_in · C_out · b_w · b_a · (1 − sparsity)

Roofline latency model
----------------------
Latency is estimated from BOPs using a simple roofline:

    latency_ms = BOPs / (peak_ops_per_ms · efficiency_factor)

The ``peak_ops_per_ms`` and ``efficiency_factor`` values are stored in a
lookup table keyed by hardware target name (``"cpu"``, ``"gpu"``, ``"fpga"``).
These are conservative approximations; the resource parser (Stage 3.5) closes
the gap by recalibrating with actual MASE compiler outputs.

Recalibration
-------------
After Stage 3.5 measures real post-compilation latency for a layer, call
``recalibrate()`` to update the efficiency factor for that hardware target.
Subsequent search iterations use the corrected model.

Usage
-----
>>> from src.hardware.cost_model import CostModel
>>> cm = CostModel(hardware_target="fpga")
>>> cm.build_from_model(model, input_size=(1, 3, 32, 32))
>>> bops = cm.compute_policy_bops(policy)
>>> latency = cm.predict_latency_ms(policy)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware lookup table
# peak_ops_per_ms: theoretical peak integer ops per millisecond
# efficiency:      fraction of peak actually achieved (roofline factor)
# ---------------------------------------------------------------------------
_HW_TABLE: Dict[str, Dict[str, float]] = {
    "cpu":  {"peak_ops_per_ms": 50_000_000,   "efficiency": 0.25},
    "gpu":  {"peak_ops_per_ms": 500_000_000,  "efficiency": 0.40},
    "fpga": {"peak_ops_per_ms": 100_000_000,  "efficiency": 0.35},
}

# Default hardware target if none specified
_DEFAULT_TARGET = "fpga"


# ---------------------------------------------------------------------------
# Layer descriptor — everything the cost model needs about one layer
# ---------------------------------------------------------------------------

class LayerSpec:
    """Static description of one layer's compute requirements.

    Attributes
    ----------
    name:       Module name (e.g. ``"layer1.0.conv1"``).
    layer_type: ``"conv"`` or ``"linear"``.
    c_in:       Input channels (or input features for Linear).
    c_out:      Output channels (or output features for Linear).
    k:          Kernel size (1 for Linear).
    h_out:      Output spatial height (1 for Linear).
    w_out:      Output spatial width (1 for Linear).
    n_params:   Number of weight parameters.
    """

    def __init__(
        self,
        name: str,
        layer_type: str,
        c_in: int,
        c_out: int,
        k: int = 1,
        h_out: int = 1,
        w_out: int = 1,
    ) -> None:
        self.name = name
        self.layer_type = layer_type
        self.c_in = c_in
        self.c_out = c_out
        self.k = k
        self.h_out = h_out
        self.w_out = w_out
        self.n_params = c_in * c_out * k * k

    def bops(self, weight_bits: int, activation_bits: int, sparsity: float = 0.0) -> float:
        """Compute BOPs for this layer under a given policy entry.

        Parameters
        ----------
        weight_bits:
            Bit-width assigned to weights.
        activation_bits:
            Bit-width assigned to activations.
        sparsity:
            Fraction of output channels pruned (0.0 = no pruning).

        Returns
        -------
        float
            Bit-operations count.
        """
        effective_c_out = self.c_out * (1.0 - sparsity)
        if self.layer_type == "conv":
            return (
                2.0 * self.c_in * effective_c_out
                * self.k * self.k
                * self.h_out * self.w_out
                * weight_bits * activation_bits
            )
        else:  # linear
            return 2.0 * self.c_in * effective_c_out * weight_bits * activation_bits


# ---------------------------------------------------------------------------
# Main cost model
# ---------------------------------------------------------------------------

class CostModel:
    """Roofline-based latency predictor for mixed-precision compressed models.

    Parameters
    ----------
    hardware_target:
        Key into ``_HW_TABLE``.  One of ``"cpu"``, ``"gpu"``, ``"fpga"``.
    """

    def __init__(self, hardware_target: str = _DEFAULT_TARGET) -> None:
        if hardware_target not in _HW_TABLE:
            raise ValueError(
                f"Unknown hardware target '{hardware_target}'. "
                f"Choose from {list(_HW_TABLE.keys())}."
            )
        self.hardware_target = hardware_target
        self._hw = dict(_HW_TABLE[hardware_target])  # mutable copy

        # layer_name -> LayerSpec
        self._specs: Dict[str, LayerSpec] = {}

        # fp32 baseline BOPs (set once after build_from_model)
        self._baseline_bops: Optional[float] = None

    # ------------------------------------------------------------------
    # Build phase — extract layer specs from a model + dummy forward
    # ------------------------------------------------------------------

    def build_from_model(
        self,
        model: nn.Module,
        input_size: Tuple[int, ...] = (1, 3, 32, 32),
        device: str = "cpu",
    ) -> None:
        """Trace the model with a dummy input to collect layer shapes.

        Parameters
        ----------
        model:
            The FP32 model (before pruning or quantization).
        input_size:
            Shape of a single input batch, e.g. ``(1, 3, 32, 32)``.
        device:
            Device to run the trace on.
        """
        model = model.to(device)
        model.eval()
        self._specs = {}

        # Hook to record output spatial dims during a dummy forward pass
        _spatial: Dict[str, Tuple[int, int]] = {}

        hooks = []

        def make_hook(name):
            def hook(module, inp, out):
                if isinstance(out, torch.Tensor) and out.dim() == 4:
                    _spatial[name] = (out.shape[2], out.shape[3])
                else:
                    _spatial[name] = (1, 1)
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            dummy = torch.zeros(input_size, device=device)
            model(dummy)

        for h in hooks:
            h.remove()

        # Build LayerSpec objects
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                h_out, w_out = _spatial.get(name, (1, 1))
                self._specs[name] = LayerSpec(
                    name=name,
                    layer_type="conv",
                    c_in=module.in_channels,
                    c_out=module.out_channels,
                    k=module.kernel_size[0],
                    h_out=h_out,
                    w_out=w_out,
                )
            elif isinstance(module, nn.Linear):
                self._specs[name] = LayerSpec(
                    name=name,
                    layer_type="linear",
                    c_in=module.in_features,
                    c_out=module.out_features,
                )

        self._baseline_bops = self._compute_bops_uniform(weight_bits=32, activation_bits=32)
        logger.info(
            "CostModel built: %d layers, baseline BOPs = %.3e",
            len(self._specs), self._baseline_bops,
        )

    # ------------------------------------------------------------------
    # Policy evaluation
    # ------------------------------------------------------------------

    def compute_policy_bops(self, policy: Dict[str, dict]) -> float:
        """Total BOPs for a compression policy.

        Parameters
        ----------
        policy:
            Dict mapping layer name to::

                {
                    "weight_bits":  int,
                    "activation_bits": int,
                    "sparsity": float   # fraction of output channels pruned
                }

            Layers not in the policy are treated as fp32 (32-bit, no pruning).

        Returns
        -------
        float
            Total BOPs across the entire model.
        """
        total = 0.0
        for name, spec in self._specs.items():
            entry = policy.get(name, {})
            wb = entry.get("weight_bits", 32)
            ab = entry.get("activation_bits", 32)
            sp = entry.get("sparsity", 0.0)
            total += spec.bops(wb, ab, sp)
        return total

    def predict_latency_ms(self, policy: Dict[str, dict]) -> float:
        """Predict inference latency in milliseconds for a policy.

        Uses the roofline model:
            latency = BOPs / (peak_ops_per_ms * efficiency)

        Parameters
        ----------
        policy:
            Same format as ``compute_policy_bops()``.

        Returns
        -------
        float
            Estimated latency in milliseconds.
        """
        bops = self.compute_policy_bops(policy)
        effective_peak = self._hw["peak_ops_per_ms"] * self._hw["efficiency"]
        return bops / effective_peak

    def bops_ratio(self, policy: Dict[str, dict]) -> float:
        """Return policy BOPs as a fraction of the fp32 baseline.

        Lower is better (more compression).

        Returns
        -------
        float
            BOPs ratio ∈ (0, 1].
        """
        if self._baseline_bops is None or self._baseline_bops == 0:
            return 1.0
        return self.compute_policy_bops(policy) / self._baseline_bops

    def get_layer_bops(
        self,
        layer_name: str,
        weight_bits: int,
        activation_bits: int,
        sparsity: float = 0.0,
    ) -> float:
        """BOPs for a single layer — used by the greedy search.

        Returns 0.0 if the layer is not in the spec (e.g. BN layers).
        """
        if layer_name not in self._specs:
            return 0.0
        return self._specs[layer_name].bops(weight_bits, activation_bits, sparsity)

    def get_param_count(self, layer_name: str) -> int:
        """Return number of weight parameters for a layer."""
        if layer_name not in self._specs:
            return 0
        return self._specs[layer_name].n_params

    def get_all_layer_names(self) -> List[str]:
        """Return all layer names that have a spec (Conv2d + Linear)."""
        return list(self._specs.keys())

    # ------------------------------------------------------------------
    # Recalibration — feedback from resource parser (Stage 3.5)
    # ------------------------------------------------------------------

    def recalibrate(
        self,
        predicted_ms: float,
        actual_ms: float,
        layer_name: Optional[str] = None,
    ) -> None:
        """Update the efficiency factor based on measured vs. predicted latency.

        Called by ``src/compiler/resource_parser.py`` after Stage 3 produces
        real hardware estimates.

        Parameters
        ----------
        predicted_ms:
            Latency predicted by this cost model before compilation.
        actual_ms:
            Actual latency measured / estimated by the MASE compiler.
        layer_name:
            Unused currently (global recalibration only).  Reserved for
            future per-layer correction.
        """
        if predicted_ms <= 0:
            logger.warning("recalibrate() called with predicted_ms <= 0, skipping.")
            return

        correction = predicted_ms / actual_ms
        old_eff = self._hw["efficiency"]
        self._hw["efficiency"] = old_eff * correction
        logger.info(
            "CostModel recalibrated: efficiency %.4f → %.4f  "
            "(predicted=%.3fms, actual=%.3fms)",
            old_eff, self._hw["efficiency"], predicted_ms, actual_ms,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_bops_uniform(self, weight_bits: int, activation_bits: int) -> float:
        """Total BOPs when all layers use the same bit-width."""
        total = 0.0
        for spec in self._specs.values():
            total += spec.bops(weight_bits, activation_bits, sparsity=0.0)
        return total
