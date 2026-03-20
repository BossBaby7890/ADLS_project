"""
src/compiler/mase_integration.py
=================================
Stage 3 of the APQ-Lite (HAWQ-lite) pipeline: **MASE/CHOP Config Generation**.

Role in the pipeline
--------------------
The MASE (Machine-learning Accelerator System Exploration) framework and its
accompanying CHOP (CHOP Hardware-Optimised Pipeline) compiler consume a
structured ``quantization_config`` dictionary to drive hardware-aware
quantization passes.  This module is the **bridge** between the abstract
bit-width assignments produced by ``src/allocator/bit_mapper.py`` and the
concrete key-value schema expected by CHOP.

CHOP quantization pass schema (per layer entry)::

    {
        "<layer_name>": {
            "weight_width":       <int>,   # total weight bits
            "weight_frac_width":  <int>,   # fractional weight bits (fixed-point)
            "data_in_width":      <int>,   # total activation bits
            "data_in_frac_width": <int>,   # fractional activation bits
            "data_out_width":     <int>,   # output activation bits (usually = data_in)
            "data_out_frac_width":<int>,
        }
    }

Fractional-bit heuristic
    ``frac_width = bit_width - 2``  (2 integer bits, rest fractional).
    This can be overridden per-layer via ``frac_overrides``.

The generated config dict can be:
  - Passed directly to a CHOP transform pass in a MASE session.
  - Written to a JSON file for offline compiler invocation.
  - Used inside ``scripts/run_qat.py`` to initialise QAT fake-quantizers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Type alias matching the allocator's output
BitMap = Dict[str, Dict[str, int]]

# CHOP-compatible per-layer config dict
ChopLayerConfig = Dict[str, int]
ChopConfig = Dict[str, ChopLayerConfig]


class MaseConfigGenerator:
    """Translate a :class:`~src.allocator.BitWidthAllocator` output into a
    MASE/CHOP-compatible ``quantization_config`` dictionary.

    Parameters
    ----------
    weight_width_key:
        CHOP key name for weight total bit-width.
    activation_width_key:
        CHOP key name for activation (data_in) total bit-width.
    weight_frac_key:
        CHOP key name for weight fractional bit-width.
    activation_frac_key:
        CHOP key name for activation fractional bit-width.
    default_frac_width:
        Fallback fractional bits when ``bit_width - 2`` would be <= 0.
    frac_overrides:
        Optional per-layer dict overriding the fractional-bit heuristic.
        E.g. ``{"layer1.conv1": {"weight_frac_width": 3}}``.
    """

    # Default CHOP key names — override via quant_params.yaml > mase_keys
    DEFAULT_WEIGHT_WIDTH_KEY = "weight_width"
    DEFAULT_ACT_WIDTH_KEY = "data_in_width"
    DEFAULT_WEIGHT_FRAC_KEY = "weight_frac_width"
    DEFAULT_ACT_FRAC_KEY = "data_in_frac_width"

    def __init__(
        self,
        weight_width_key: str = DEFAULT_WEIGHT_WIDTH_KEY,
        activation_width_key: str = DEFAULT_ACT_WIDTH_KEY,
        weight_frac_key: str = DEFAULT_WEIGHT_FRAC_KEY,
        activation_frac_key: str = DEFAULT_ACT_FRAC_KEY,
        default_frac_width: int = 6,
        frac_overrides: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> None:
        self.weight_width_key = weight_width_key
        self.activation_width_key = activation_width_key
        self.weight_frac_key = weight_frac_key
        self.activation_frac_key = activation_frac_key
        self.default_frac_width = default_frac_width
        self.frac_overrides = frac_overrides or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, bit_map: BitMap) -> ChopConfig:
        """Convert a bit-width allocation map into a CHOP quantization config.

        Parameters
        ----------
        bit_map:
            Output of :meth:`~src.allocator.BitWidthAllocator.allocate`.
            Expected structure::

                {
                    "layer_name": {"weight_bits": 4, "activation_bits": 4},
                    ...
                }

        Returns
        -------
        ChopConfig
            A dict ready to be passed to a MASE/CHOP quantization pass.
        """
        chop_config: ChopConfig = {}

        for layer_name, cfg in bit_map.items():
            w_bits = cfg["weight_bits"]
            a_bits = cfg["activation_bits"]

            w_frac = self._frac_bits(w_bits, layer_name, "weight")
            a_frac = self._frac_bits(a_bits, layer_name, "activation")

            layer_config: ChopLayerConfig = {
                self.weight_width_key: w_bits,
                self.weight_frac_key: w_frac,
                self.activation_width_key: a_bits,
                self.activation_frac_key: a_frac,
                # data_out mirrors data_in (standard assumption)
                "data_out_width": a_bits,
                "data_out_frac_width": a_frac,
            }

            # Merge any per-layer overrides last
            layer_config.update(self.frac_overrides.get(layer_name, {}))
            chop_config[layer_name] = layer_config

            logger.debug(
                "  %-40s  W=%d/%d  A=%d/%d",
                layer_name,
                w_bits,
                w_frac,
                a_bits,
                a_frac,
            )

        logger.info(
            "Generated CHOP quantization config for %d layers.", len(chop_config)
        )
        return chop_config

    def save(self, chop_config: ChopConfig, output_path: str | Path) -> None:
        """Serialise the quantization config to a JSON file.

        The JSON file can be loaded by the MASE CLI or Python API::

            from chop.passes.graph.transforms import quantize_transform_pass
            quantize_transform_pass(mg, config=json.load(open("quant_config.json")))

        Parameters
        ----------
        chop_config:
            Output of :meth:`generate`.
        output_path:
            Destination path (will be created if necessary).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(chop_config, fh, indent=2)
        logger.info("CHOP quantization config saved to %s", output_path)

    @staticmethod
    def load(config_path: str | Path) -> ChopConfig:
        """Load a previously saved CHOP quantization config from JSON."""
        with open(config_path) as fh:
            return json.load(fh)

    def wrap_for_mase_pass(
        self,
        chop_config: ChopConfig,
        pass_name: str = "quantize",
    ) -> Dict[str, Any]:
        """Wrap the flat config in the top-level dict expected by MASE passes.

        MASE transform passes typically expect::

            {
                "by": "name",
                "default": {...},
                "<layer_name>": {...},
            }

        Parameters
        ----------
        chop_config:
            Per-layer config from :meth:`generate`.
        pass_name:
            MASE pass identifier string (informational only).

        Returns
        -------
        dict
            Ready-to-use MASE pass argument dict.
        """
        mase_pass_config: Dict[str, Any] = {
            "by": "name",
            "default": {
                self.weight_width_key: 8,
                self.weight_frac_key: 6,
                self.activation_width_key: 8,
                self.activation_frac_key: 6,
                "data_out_width": 8,
                "data_out_frac_width": 6,
            },
        }
        mase_pass_config.update(chop_config)
        logger.debug("Wrapped config for MASE pass '%s'.", pass_name)
        return mase_pass_config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _frac_bits(self, total_bits: int, layer_name: str, kind: str) -> int:
        """Compute fractional bit-width using the ``bit_width - 2`` heuristic."""
        override_key = f"{kind}_frac_width"
        override = self.frac_overrides.get(layer_name, {}).get(override_key)
        if override is not None:
            return override
        frac = total_bits - 2
        return max(frac, self.default_frac_width) if frac > 0 else self.default_frac_width
