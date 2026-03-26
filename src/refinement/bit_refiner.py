from __future__ import annotations

import copy
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

BitMap = Dict[str, Dict[str, int]]


class BitWidthRefiner:
    """
    Post-allocation bit-map refinement.

    Starting from a threshold-based mixed-precision allocation, this module
    upgrades a small number of highly sensitive low-bit layers to improve
    robustness before AdaRound.
    """

    def __init__(
        self,
        low_bit: int = 2,
        rescue_bit: int = 4,
        max_rescues: int = 3,
        min_sensitivity_for_rescue: float = 0.05,
        protect_patterns: List[str] | None = None,
    ) -> None:
        self.low_bit = low_bit
        self.rescue_bit = rescue_bit
        self.max_rescues = max_rescues
        self.min_sensitivity_for_rescue = min_sensitivity_for_rescue
        self.protect_patterns = protect_patterns or []

    def refine(
        self,
        bit_map: BitMap,
        sensitivity_scores: Dict[str, float],
    ) -> Tuple[BitMap, List[str]]:
        refined = copy.deepcopy(bit_map)

        candidates = []
        for layer_name, cfg in refined.items():
            if cfg["weight_bits"] != self.low_bit:
                continue

            if any(pat in layer_name for pat in self.protect_patterns):
                continue

            # Restrict refinement to actual quantizable weight-bearing layers
            if not (
                layer_name.endswith("conv1")
                or layer_name.endswith("conv2")
                or layer_name == "linear"
                or ".shortcut.0" in layer_name
            ):
                continue

            sens = float(sensitivity_scores.get(layer_name, 0.0))
            if sens >= self.min_sensitivity_for_rescue:
                candidates.append((layer_name, sens))

            

        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = candidates[: self.max_rescues]

        rescued_layers = []
        for layer_name, sens in selected:
            refined[layer_name]["weight_bits"] = self.rescue_bit
            refined[layer_name]["activation_bits"] = min(
                self.rescue_bit,
                refined[layer_name].get("activation_bits", self.rescue_bit),
            )
            rescued_layers.append(layer_name)

        if rescued_layers:
            logger.info(
                "Bit refinement rescued %d layer(s): %s",
                len(rescued_layers),
                ", ".join(rescued_layers),
            )
        else:
            logger.info("Bit refinement made no changes.")

        return refined, rescued_layers

    @staticmethod
    def summarise_changes(
        original: BitMap,
        refined: BitMap,
    ) -> Dict[str, int]:
        summary = {
            "changed_layers": 0,
            "upgraded_2_to_4": 0,
            "upgraded_4_to_8": 0,
            "downgraded_8_to_4": 0,
            "downgraded_4_to_2": 0,
        }

        for layer_name in original:
            if layer_name not in refined:
                continue

            old_w = original[layer_name]["weight_bits"]
            new_w = refined[layer_name]["weight_bits"]

            if old_w != new_w:
                summary["changed_layers"] += 1
                if old_w == 2 and new_w == 4:
                    summary["upgraded_2_to_4"] += 1
                elif old_w == 4 and new_w == 8:
                    summary["upgraded_4_to_8"] += 1
                elif old_w == 8 and new_w == 4:
                    summary["downgraded_8_to_4"] += 1
                elif old_w == 4 and new_w == 2:
                    summary["downgraded_4_to_2"] += 1

        return summary

