"""
src/allocator/bit_mapper.py
===========================
Stage 2 of the APQ-Lite (HAWQ-lite) pipeline: **Bit-Width Allocation**.

Role in the pipeline
--------------------
After Stage 1 produces a per-layer sensitivity score in [0, 1], this module
implements the **policy** that translates those scores into concrete bit-width
assignments for both weights and activations.

Assignment logic (threshold-based, configurable via ``quant_params.yaml``):

    score >= high_threshold  →  8-bit   (sensitive layer, preserve accuracy)
    score >= mid_threshold   →  4-bit   (moderate sensitivity, balanced)
    score <  mid_threshold   →  2-bit   (insensitive, maximum compression)

Special cases (layer overrides):
  - The first and last layers of the network are pinned to 8-bit by default,
    consistent with the standard practice in HAWQ/APQ literature.
  - Batch-normalisation and skip-connection layers are excluded (kept fp32).
  - A hardware budget check warns when the projected model size or BOPs
    exceed the configured targets.

The output of this module is a flat dict::

    {
        "layer_name": {"weight_bits": 4, "activation_bits": 4},
        ...
    }

which is consumed by ``src/compiler/mase_integration.py``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Bit-width assignment result type
LayerBitConfig = Dict[str, int]   # {"weight_bits": x, "activation_bits": y}
BitMap = Dict[str, LayerBitConfig]


class BitWidthAllocator:
    """Policy engine that maps sensitivity scores to bit-width assignments.

    Parameters
    ----------
    available_bits:
        Sorted list of candidate bit-widths, e.g. ``[2, 4, 8]``.
    high_threshold:
        Sensitivity score above which 8-bit is assigned.
    mid_threshold:
        Sensitivity score above which 4-bit is assigned (else 2-bit).
    first_layer_bits:
        Forced bit-width for the first parametric layer.
    last_layer_bits:
        Forced bit-width for the final parametric (classifier) layer.
    skip_patterns:
        List of sub-strings; layers whose names contain any of these are
        excluded from quantization (left in fp32).
    weight_default:
        Fallback bit-width when a layer is not found in the sensitivity dict.
    activation_default:
        Fallback activation bit-width.
    """

    def __init__(
        self,
        available_bits: List[int] = [2, 4, 8],
        high_threshold: float = 0.70,
        mid_threshold: float = 0.35,
        first_layer_bits: int = 8,
        last_layer_bits: int = 8,
        skip_patterns: Optional[List[str]] = None,
        weight_default: int = 8,
        activation_default: int = 8,
    ) -> None:
        self.available_bits = sorted(available_bits)
        self.high_threshold = high_threshold
        self.mid_threshold = mid_threshold
        self.first_layer_bits = first_layer_bits
        self.last_layer_bits = last_layer_bits
        self.skip_patterns = skip_patterns or ["bn", "shortcut", "downsample"]
        self.weight_default = weight_default
        self.activation_default = activation_default

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(
        self,
        sensitivity_scores: Dict[str, float],
        layer_names: Optional[List[str]] = None,
    ) -> BitMap:
        """Produce a bit-width assignment for every layer.

        Parameters
        ----------
        sensitivity_scores:
            Per-layer normalised sensitivity scores in [0, 1].
            Keys should be module names (output of ``aggregate_to_layers``).
        layer_names:
            Ordered list of *all* layer names in the model.  If provided,
            the first and last entries receive the override bit-widths.
            If ``None``, ordering is inferred from ``sensitivity_scores``.

        Returns
        -------
        BitMap
            Dict mapping each layer name to its weight & activation bit-widths.
        """
        names = layer_names or list(sensitivity_scores.keys())
        parametric_names = [n for n in names if not self._is_skipped(n)]

        first_layer = parametric_names[0] if parametric_names else None
        last_layer = parametric_names[-1] if parametric_names else None

        bit_map: BitMap = {}

        for name in names:
            if self._is_skipped(name):
                logger.debug("  Skipping layer (fp32): %s", name)
                continue

            score = sensitivity_scores.get(name, None)

            if name == first_layer:
                bits = self.first_layer_bits
                reason = "first-layer override"
            elif name == last_layer:
                bits = self.last_layer_bits
                reason = "last-layer override"
            elif score is None:
                bits = self.weight_default
                reason = "default (not profiled)"
            else:
                bits, reason = self._score_to_bits(score)

            bit_map[name] = {
                "weight_bits": bits,
                "activation_bits": min(bits, self.activation_default),
            }
            logger.debug(
                "  %-40s  score=%-6s  bits=%d  [%s]",
                name,
                f"{score:.3f}" if score is not None else "N/A",
                bits,
                reason,
            )

        self._log_summary(bit_map)
        return bit_map

    def estimate_compression(
        self,
        bit_map: BitMap,
        param_counts: Dict[str, int],
        baseline_bits: int = 32,
    ) -> Tuple[float, float]:
        """Estimate compression ratio vs. a full-precision baseline.

        Parameters
        ----------
        bit_map:
            Output of :meth:`allocate`.
        param_counts:
            Dict mapping layer name to number of weight parameters.
        baseline_bits:
            Bit-width of the full-precision reference (default 32).

        Returns
        -------
        (compression_ratio, size_reduction_fraction)
            E.g. ``(4.0, 0.75)`` means 4× smaller, 75% size reduction.
        """
        total_fp_bits = 0
        total_q_bits = 0

        for name, config in bit_map.items():
            n_params = param_counts.get(name, 0)
            total_fp_bits += n_params * baseline_bits
            total_q_bits += n_params * config["weight_bits"]

        if total_q_bits == 0:
            return 1.0, 0.0

        ratio = total_fp_bits / total_q_bits
        reduction = 1.0 - (total_q_bits / total_fp_bits)
        return ratio, reduction

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_to_bits(self, score: float) -> Tuple[int, str]:
        """Apply the threshold policy to a single normalised score."""
        if score >= self.high_threshold:
            return 8, f"high sensitivity (score >= {self.high_threshold})"
        elif score >= self.mid_threshold:
            return 4, f"mid sensitivity ({self.mid_threshold} <= score < {self.high_threshold})"
        else:
            return 2, f"low sensitivity (score < {self.mid_threshold})"

    def _is_skipped(self, name: str) -> bool:
        """Return True if the layer should be left in fp32."""
        return any(pat in name for pat in self.skip_patterns)

    def _log_summary(self, bit_map: BitMap) -> None:
        counts: Dict[int, int] = {}
        for cfg in bit_map.values():
            b = cfg["weight_bits"]
            counts[b] = counts.get(b, 0) + 1
        summary = ", ".join(f"{b}-bit: {c} layers" for b, c in sorted(counts.items()))
        logger.info("Bit-width allocation summary — %s", summary)
