#!/usr/bin/env python3
"""
scripts/export_onnx.py
======================
Entry-point for **Stage 4**: ONNX Export for Hardware Deployment.

Loads the QAT-trained (or converted) model and exports it to ONNX format,
optionally running a basic shape-correctness check via ``onnxruntime``.

The exported ``.onnx`` file can be:
  - Fed into ``onnxsim`` for graph simplification.
  - Compiled with TVM / MASE HLS backend for FPGA/ASIC targets.
  - Evaluated with ``onnxruntime`` for CPU/GPU latency benchmarking.

Usage
-----
    python scripts/export_onnx.py --config configs/base_config.yaml \\
                                   --checkpoint outputs/checkpoints/checkpoint_best.pth \\
                                   --output    outputs/model_quantized.onnx \\
                                   --opset     17
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger("export_onnx")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def verify_onnx(onnx_path: str, dummy_input_shape: tuple) -> bool:
    """Run a forward pass with onnxruntime to verify the exported graph."""
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        dummy = np.random.randn(*dummy_input_shape).astype(np.float32)
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: dummy})
        logger.info(
            "ONNX verification passed. Output shape: %s", outputs[0].shape
        )
        return True
    except ImportError:
        logger.warning("onnxruntime not installed — skipping verification.")
        return False
    except Exception as exc:
        logger.error("ONNX verification failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="APQ-Lite — ONNX Export")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--output",
        default="outputs/model_quantized.onnx",
        help="Destination path for the exported ONNX file.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip onnxruntime verification after export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(args.device)
    logger.info("Export device: %s", device)

    # ---- Build and load model ----
    num_classes = 10 if cfg["data"]["dataset"] == "cifar10" else 100
    model = build_model(cfg["model"]["architecture"], num_classes=num_classes)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.to(device).eval()
    logger.info("Checkpoint loaded from %s", args.checkpoint)

    # ---- Dummy input ----
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, 32, 32, device=device)

    # ---- Export ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting to ONNX (opset=%d) → %s", args.opset, output_path
    )
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=args.opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        do_constant_folding=True,
        export_params=True,
    )
    logger.info("Export complete: %s  (%.2f MB)", output_path, output_path.stat().st_size / 1e6)

    # ---- Optional verification ----
    if not args.no_verify:
        verify_onnx(str(output_path), (batch_size, 3, 32, 32))


if __name__ == "__main__":
    main()
