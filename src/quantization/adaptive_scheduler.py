from __future__ import annotations

from typing import Dict, Any


class AdaptiveAdaRoundScheduler:
    """
    Per-layer AdaRound budget allocator based on:
      - assigned bit-width
      - sensitivity score

    This is an additive module. It does not modify the base AdaRound algorithm.
    """

    def __init__(
        self,
        base_steps: int = 500,
        min_steps: int = 100,
        max_steps: int = 1500,
        sensitivity_scale: float = 1.0,
        low_bit_boost: float = 1.5,
        mid_bit_boost: float = 1.0,
        high_bit_boost: float = 0.5,
        base_calib_batches: int = 1,
        max_calib_batches: int = 4,
    ) -> None:
        self.base_steps = base_steps
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.sensitivity_scale = sensitivity_scale
        self.low_bit_boost = low_bit_boost
        self.mid_bit_boost = mid_bit_boost
        self.high_bit_boost = high_bit_boost
        self.base_calib_batches = base_calib_batches
        self.max_calib_batches = max_calib_batches

    def get_layer_budget(
        self,
        layer_name: str,
        bit_width: int,
        sensitivity: float | None,
    ) -> Dict[str, Any]:
        s = 0.0 if sensitivity is None else float(sensitivity)

        if bit_width <= 2:
            bit_factor = self.low_bit_boost
        elif bit_width <= 4:
            bit_factor = self.mid_bit_boost
        else:
            bit_factor = self.high_bit_boost

        steps = int(self.base_steps * bit_factor * (1.0 + self.sensitivity_scale * s))
        steps = max(self.min_steps, min(self.max_steps, steps))

        calib_batches = self.base_calib_batches
        if s >= 0.8:
            calib_batches = min(self.max_calib_batches, self.base_calib_batches + 2)
        elif s >= 0.5:
            calib_batches = min(self.max_calib_batches, self.base_calib_batches + 1)

        return {
            "layer_name": layer_name,
            "steps": steps,
            "calib_batches": calib_batches,
            "bit_width": bit_width,
            "sensitivity": s,
        }

    def build_schedule(
        self,
        bit_map: Dict[str, Dict[str, int]],
        sensitivity_scores: Dict[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        schedule: Dict[str, Dict[str, Any]] = {}
        for layer_name, cfg in bit_map.items():
            bit_width = int(cfg["weight_bits"])
            sensitivity = sensitivity_scores.get(layer_name, 0.0)
            schedule[layer_name] = self.get_layer_budget(
                layer_name=layer_name,
                bit_width=bit_width,
                sensitivity=sensitivity,
            )
        return schedule

