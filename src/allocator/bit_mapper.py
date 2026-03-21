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


## NEWLY ADDED DICTIONARIES FOR COSTS

#Estimation of a cost factor to each bit width
BIT_COST: Dict[int, float] = {
    8: 1.00,
    4: 0.55,
    2: 0.30,
}


# Instead of directly going down to a lower width(which reduces costs *significantly*), it goes down
# to per layer from per channel (which is computationally less expensive than per channel instead)
GRANULARITY_COST: Dict[str, float] = {
    "per_channel": 1.15,
    "per_layer":   1.00,
}

# each downgrading step
DOWNGRADE_LADDER: List[Tuple[int, str]] = [
    (8, "per_channel"),
    (8, "per_layer"),
    (4, "per_channel"),
    (4, "per_layer"),
    (2, "per_layer"),  ## worst quality
]


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

    # def __init__(
    #     self,
    #     available_bits: List[int] = [2, 4, 8],
    #     high_threshold: float = 0.70,
    #     mid_threshold: float = 0.35,
    #     first_layer_bits: int = 8,
    #     last_layer_bits: int = 8,
    #     skip_patterns: Optional[List[str]] = None,
    #     weight_default: int = 8,
    #     activation_default: int = 8,
    # ) -> None:
    #     self.available_bits = sorted(available_bits)
    #     self.high_threshold = high_threshold
    #     self.mid_threshold = mid_threshold
    #     self.first_layer_bits = first_layer_bits
    #     self.last_layer_bits = last_layer_bits
    #     self.skip_patterns = skip_patterns or ["bn", "shortcut", "downsample"]
    #     self.weight_default = weight_default
    #     self.activation_default = activation_default

    ## changed the parameters for the greedy estimation
    def __init__(
        self,
        available_bits: List[int] = [2, 4, 8],
        target_compression_ratio: Optional[float] = 2.0, # the quantized model should use half the bits of 8bit model. (basically there is small compression)
        first_layer_bits: int = 8,
        last_layer_bits: int = 8,
        skip_patterns: Optional[List[str]] = None,
        activation_default: int = 8,
    ) -> None:
        self.available_bits = sorted(available_bits)
        self.target_compression_ratio = target_compression_ratio 
        self.first_layer_bits = first_layer_bits
        self.last_layer_bits = last_layer_bits
        self.skip_patterns = skip_patterns or ["bn", "shortcut", "downsample"]
        self.activation_default = activation_default

        # setting floor on the bit assignment
         self._ladder = [
            step for step in DOWNGRADE_LADDER
            if step[0] in self.available_bits
        ]
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # def allocate(
    #     self,
    #     sensitivity_scores: Dict[str, float],
    #     layer_names: Optional[List[str]] = None,
    # ) -> BitMap:

    def allocate(
        self,
        sensitivity_scores: Dict[str, float],
        param_counts: Dict[str, int], 
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

        if not parametric_names:
            logger.warning("No quantizable layers found — returning empty BitMap.")
            return {}

        first_layer = parametric_names[0]
        last_layer  = parametric_names[-1]
        #first_layer = parametric_names[0] if parametric_names else None
        #last_layer = parametric_names[-1] if parametric_names else None

        #bit_map: BitMap = {}

        # Tracks where every layers sits on the ladder
        ladder_idx: Dict[str, int] = {}
        for name in parametric_names:
            ladder_idx[name] = 0 

        #locking first and last layers 
        locked: set = set()
        locked.add(first_layer)
        locked.add(last_layer)

        # computing the base cost
        current_cost  = self._compute_total_cost(ladder_idx, param_counts, locked,
                                                  first_layer, last_layer)
        baseline_cost = self._compute_baseline_cost(param_counts, locked,
                                                     first_layer, last_layer)


        # Deriving budget from the target compression ratio
        if self.target_compression_ratio is None:
            budget = current_cost   
        else:
            budget = baseline_cost / self.target_compression_ratio

        logger.info(
            "Greedy search - baseline cost: %.2f  budget: %.2f  "
            "(target ratio: %sx)",
            baseline_cost, budget,
            f"{self.target_compression_ratio:.1f}" if self.target_compression_ratio else "none",
        )
        
        # the greedy downgrading loop 
        iteration = 0
        while current_cost > budget:
            # Find the nonlocked layer with the lowest sensitivity score
            # that has not yet reached the bottom of the ladder.
            candidate = self._pick_candidate(
                ladder_idx, sensitivity_scores, locked
            )
 
            if candidate is None:
                logger.warning(
                    "All layers are at minimum precision but cost (%.2f) "
                    "is still > budget (%.2f).  "
                    "Consider relaxing the compression target.",
                    current_cost, budget,
                )
                break
 
            # downgrading the chosen layer by one step
            old_idx = ladder_idx[candidate]
            ladder_idx[candidate] = old_idx + 1
            new_bits, new_gran = self._ladder[ladder_idx[candidate]]
 
            current_cost = self._compute_total_cost(
                ladder_idx, param_counts, locked, first_layer, last_layer
            )
 
            logger.debug(
                "  iter %03d | downgrade %-40s  %s→%s  "
                "cost=%.4f  budget=%.4f",
                iteration,
                candidate,
                self._ladder[old_idx],
                (new_bits, new_gran),
                current_cost,
                budget,
            )
            iteration += 1
 
        logger.info(
            "Greedy search complete in %d iterations.  "
            "Final cost: %.4f  Budget: %.4f  Achieved ratio: %.2f×",
            iteration, current_cost, budget,
            baseline_cost / current_cost if current_cost > 0 else float("inf"),
        )
        bit_map: BitMap = {}

        for name in parametric_names:
            if name == first_layer:
                w_bits  = self.first_layer_bits
                gran    = "per_channel"
                reason  = "first-layer override"
            elif name == last_layer:
                w_bits  = self.last_layer_bits
                gran    = "per_channel"
                reason  = "last-layer override"
            else:
                w_bits, gran = self._ladder[ladder_idx[name]]
                reason = f"greedy (ladder idx {ladder_idx[name]})"

            a_bits = min(w_bits, self.activation_default)
 
            bit_map[name] = {
                "weight_bits":      w_bits,
                "activation_bits":  a_bits,
                "granularity":      gran,
            }
 
            score = sensitivity_scores.get(name)
            logger.debug(
                "  %-40s  score=%-6s  w=%d-bit  a=%d-bit  gran=%-11s  [%s]",
                name,
                f"{score:.3f}" if score is not None else "N/A",
                w_bits, a_bits, gran, reason,
            )
        self._log_summary(bit_map)
        return bit_map

        # pareto = sweeping across many targets instead of just 1 compression target, gives learning curve for 
        # debugging and checking purposes
        # straight from HAWQ-V2 / APQ
        def allocate_pareto(
            self,
            sensitivity_scores: Dict[str, float],
            param_counts: Dict[str, int],
            compression_ratios: List[float],
            layer_names: Optional[List[str]] = None,
        ) -> List[Tuple[float, BitMap]]:


        results = []
        for ratio in compression_ratios:
            logger.info("Pareto sweep — target ratio: %.2f×", ratio)
            self.target_compression_ratio = ratio
            bmap = self.allocate(sensitivity_scores, param_counts, layer_names)
            achieved = self._achieved_ratio(bmap, param_counts)
            results.append((achieved, bmap))
            logger.info("  Achieved ratio: %.3f×", achieved)
        return results


        # for name in names:
        #     if self._is_skipped(name):
        #         logger.debug("  Skipping layer (fp32): %s", name)
        #         continue

        #     score = sensitivity_scores.get(name, None)

        #     if name == first_layer:
        #         bits = self.first_layer_bits
        #         reason = "first-layer override"
        #     elif name == last_layer:
        #         bits = self.last_layer_bits
        #         reason = "last-layer override"
        #     elif score is None:
        #         bits = self.weight_default
        #         reason = "default (not profiled)"
        #     else:
        #         bits, reason = self._score_to_bits(score)

        #     bit_map[name] = {
        #         "weight_bits": bits,
        #         "activation_bits": min(bits, self.activation_default),
        #     }
        #     logger.debug(
        #         "  %-40s  score=%-6s  bits=%d  [%s]",
        #         name,
        #         f"{score:.3f}" if score is not None else "N/A",
        #         bits,
        #         reason,
        #     )

        # self._log_summary(bit_map)
        # return bit_map

    def estimate_compression(
        self,
        bit_map: BitMap,
        param_counts: Dict[str, int],
        #baseline_bits: int = 32,
        baseline_bits: int = 8, #changed for greedy comparing it to uniform 8 bit quant model
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
        #total_fp_bits = 0

        #calculating total quantized same as previous scriptn but different methodology
        total_q_bits = 0
        
        total_baseline = sum(
            param_counts.get(n, 0) * baseline_bits for n in bit_map
        )
        total_q_bits = sum(
            param_counts.get(n, 0) * cfg["weight_bits"] for n, cfg in bit_map.items()
        )


        # for name, config in bit_map.items():
        #     n_params = param_counts.get(name, 0)
        #     total_fp_bits += n_params * baseline_bits
        #     total_q_bits += n_params * config["weight_bits"]

        if total_q_bits == 0:
            return 1.0, 0.0

        # ratio = total_fp_bits / total_q_bits
        ratio = total_baseline / total_q_bits

        reduction = 1.0 - (total_q_bits / total_baseline)
        return ratio, reduction
    
    def _layer_cost(
        self,
        n_params: int,
        bits: int,
        granularity: str,
    ) -> float:
        return n_params * BIT_COST[bits] * GRANULARITY_COST[granularity]
    
    def _compute_total_cost(
        self,
        ladder_idx: Dict[str, int],
        param_counts: Dict[str, int],
        locked: set,
        first_layer: str,
        last_layer: str,
    ) -> float:
        total = 0.0
        for name, idx in ladder_idx.items():
            n = param_counts.get(name, 0)
            if name in locked:
                if name == first_layer:
                    bits, gran = self.first_layer_bits, "per_channel"
                else:
                    bits, gran = self.last_layer_bits, "per_channel"
            else:
                bits, gran = self._ladder[idx]
            total += self._layer_cost(n, bits, gran)
        return total
 
    def _compute_baseline_cost(
        self,
        param_counts: Dict[str, int],
        locked: set,
        first_layer: str,
        last_layer: str,
    ) -> float:
        """Baseline = every layer at index 0 (8-bit per-channel)."""
        bits, gran = self._ladder[0]   # (8, "per_channel")
        total = 0.0
        for name, n in param_counts.items():
            if self._is_skipped(name):
                continue
            total += self._layer_cost(n, bits, gran)
        return total
 
    def _achieved_ratio(
        self,
        bit_map: BitMap,
        param_counts: Dict[str, int],
    ) -> float:
        baseline = sum(
            param_counts.get(n, 0) * BIT_COST[8] * GRANULARITY_COST["per_channel"]
            for n in bit_map
        )
        actual = sum(
            param_counts.get(n, 0)
            * BIT_COST[cfg["weight_bits"]]
            * GRANULARITY_COST[cfg["granularity"]]
            for n, cfg in bit_map.items()
        )
        return baseline / actual if actual > 0 else 1.0
    

    ### GREEDY HELPER 

    def _pick_candidate(
        self,
        ladder_idx: Dict[str, int],
        sensitivity_scores: Dict[str, float],
        locked: set,
    ) -> Optional[str]:
        """Return the non-locked, downgradeable layer with lowest sensitivity.
 
        Among layers that are tied on sensitivity, prefer the one that is
        currently at a higher (more expensive) ladder position.
        """
        best_name  = None
        best_score = float("inf")
        best_idx   = -1
 
        max_idx = len(self._ladder) - 1
 
        for name, idx in ladder_idx.items():
            if name in locked:
                continue
            if idx >= max_idx:
                continue   # already at minimum — cannot downgrade further
 
            score = sensitivity_scores.get(name, 0.0)
 
            # Lower sensitivity wins; break ties by preferring higher ladder idx
            if score < best_score or (score == best_score and idx > best_idx):
                best_name  = name
                best_score = score
                best_idx   = idx
 
        return best_name

    ### is_skipped helper
    def _is_skipped(self, name: str) -> bool:
        """Return True if the layer should be left in fp32."""
        return any(pat in name for pat in self.skip_patterns)
 
    def _log_summary(self, bit_map: BitMap) -> None:
        counts: Dict[str, int] = {}
        for cfg in bit_map.values():
            key = f"{cfg['weight_bits']}-bit {cfg['granularity']}"
            counts[key] = counts.get(key, 0) + 1
        summary = ", ".join(
            f"{k}: {v} layers" for k, v in sorted(counts.items())
        )
        logger.info("Bit-width allocation summary — %s", summary)
 


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # def _score_to_bits(self, score: float) -> Tuple[int, str]:
    #     """Apply the threshold policy to a single normalised score."""
    #     if score >= self.high_threshold:
    #         return 8, f"high sensitivity (score >= {self.high_threshold})"
    #     elif score >= self.mid_threshold:
    #         return 4, f"mid sensitivity ({self.mid_threshold} <= score < {self.high_threshold})"
    #     else:
    #         return 2, f"low sensitivity (score < {self.mid_threshold})"

    # def _is_skipped(self, name: str) -> bool:
    #     """Return True if the layer should be left in fp32."""
    #     return any(pat in name for pat in self.skip_patterns)

    # def _log_summary(self, bit_map: BitMap) -> None:
    #     counts: Dict[int, int] = {}
    #     for cfg in bit_map.values():
    #         b = cfg["weight_bits"]
    #         counts[b] = counts.get(b, 0) + 1
    #     summary = ", ".join(f"{b}-bit: {c} layers" for b, c in sorted(counts.items()))
    #     logger.info("Bit-width allocation summary — %s", summary)
