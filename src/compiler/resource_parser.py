"""
src/compiler/resource_parser.py
================================
Stage 3.5 — **MASE Resource Parser and Cost Model Feedback**.

Role in the pipeline
--------------------
After Stage 3 runs ``quantize_transform_pass`` and produces a compiled
checkpoint, the MASE / CHOP compiler can optionally emit hardware resource
estimates (for FPGA targets: LUT count, DSP blocks, BRAM usage, estimated
latency).  This module:

  1. Parses those estimates from the MASE output directory.
  2. Compares them against the cost model's predictions.
  3. Calls ``cost_model.recalibrate()`` so subsequent search iterations
     use corrected latency estimates.
  4. Returns a structured summary for logging and notebook visualisation.

MASE output format
------------------
MASE / CHOP currently writes hardware estimates in JSON when the
``--emit-hardware-report`` flag is passed to the compiler pass.  The
expected schema is::

    {
        "total_latency_ms": 12.4,
        "layers": {
            "layer1.0.conv1": {"latency_ms": 1.2, "luts": 1024, "dsps": 8},
            ...
        }
    }

If this file is not found (e.g. when running without FPGA toolchain),
the parser falls back to a no-op and logs a warning — the pipeline
continues without recalibration.

Usage
-----
>>> from src.compiler.resource_parser import ResourceParser
>>> parser = ResourceParser(cost_model, mase_output_dir="outputs/")
>>> summary = parser.parse_and_recalibrate(predicted_latency_ms=12.0)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from src.hardware.cost_model import CostModel

logger = logging.getLogger(__name__)

# Filename that MASE emits hardware estimates to
_MASE_HW_REPORT = "hardware_report.json"


class ResourceParser:
    """Parse MASE compiler hardware estimates and recalibrate the cost model.

    Parameters
    ----------
    cost_model:
        The ``CostModel`` instance used during search.  Will be recalibrated
        in-place if a hardware report is found.
    mase_output_dir:
        Directory where MASE writes its outputs (``outputs/`` by default).
    """

    def __init__(
        self,
        cost_model: CostModel,
        mase_output_dir: str | Path = "outputs/",
    ) -> None:
        self.cost_model = cost_model
        self.output_dir = Path(mase_output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_and_recalibrate(
        self,
        predicted_latency_ms: float,
        report_filename: str = _MASE_HW_REPORT,
    ) -> Optional[Dict]:
        """Parse MASE hardware report and recalibrate cost model.

        Parameters
        ----------
        predicted_latency_ms:
            Latency predicted by the cost model *before* compilation.
            Used to compute the correction factor.
        report_filename:
            Name of the hardware report file in ``mase_output_dir``.

        Returns
        -------
        Dict with parsed hardware estimates, or ``None`` if report not found.
        """
        report_path = self.output_dir / report_filename

        if not report_path.exists():
            logger.warning(
                "MASE hardware report not found at %s. "
                "Cost model will not be recalibrated for this iteration. "
                "Pass --emit-hardware-report to the MASE compiler to enable.",
                report_path,
            )
            return None

        with open(report_path) as fh:
            report = json.load(fh)

        actual_latency_ms = report.get("total_latency_ms")
        if actual_latency_ms is None:
            logger.warning(
                "Hardware report found but 'total_latency_ms' key missing. "
                "Skipping recalibration."
            )
            return report

        logger.info(
            "Hardware report parsed — predicted=%.3fms  actual=%.3fms",
            predicted_latency_ms, actual_latency_ms,
        )

        self.cost_model.recalibrate(
            predicted_ms=predicted_latency_ms,
            actual_ms=actual_latency_ms,
        )

        summary = self._build_summary(report, predicted_latency_ms)
        self._log_summary(summary)
        return summary

    def save_summary(self, summary: Dict, output_path: str | Path) -> None:
        """Write the resource summary to JSON for notebook visualisation."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("Resource summary saved to %s", output_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_summary(
        self, report: Dict, predicted_latency_ms: float
    ) -> Dict:
        """Build a structured summary dict for logging and export."""
        actual_ms = report.get("total_latency_ms", 0.0)
        layers = report.get("layers", {})

        total_luts = sum(v.get("luts", 0) for v in layers.values())
        total_dsps = sum(v.get("dsps", 0) for v in layers.values())

        return {
            "predicted_latency_ms": predicted_latency_ms,
            "actual_latency_ms": actual_ms,
            "latency_error_pct": abs(predicted_latency_ms - actual_ms)
            / max(actual_ms, 1e-6)
            * 100,
            "total_luts": total_luts,
            "total_dsps": total_dsps,
            "per_layer": layers,
            "hw_target": self.cost_model.hardware_target,
            "recalibrated_efficiency": self.cost_model._hw["efficiency"],
        }

    def _log_summary(self, summary: Dict) -> None:
        logger.info(
            "Resource summary | LUTs=%d  DSPs=%d  "
            "latency_error=%.1f%%  recalibrated_efficiency=%.4f",
            summary["total_luts"],
            summary["total_dsps"],
            summary["latency_error_pct"],
            summary["recalibrated_efficiency"],
        )
