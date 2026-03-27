#!/usr/bin/env python3
"""
scripts/export_onnx.py
======================
Entry-point for **Stage 4**: ONNX Export for Hardware Deployment.

Loads the AdaRounded + CHOP-quantized model produced by ``run_qat.py``
(Stage 3) and exports it to ONNX format, optionally running a basic
shape-correctness check via ``onnxruntime``.

Pipeline position
-----------------
This script is the final stage of the HA-AdaRound pipeline:

    run_profiling.py   →  outputs/layer_sensitivity.json       (Stage 1)
    run_qat.py         →  outputs/checkpoints/checkpoint_quantized.pth
                          outputs/quant_config.json             (Stage 2/2.5/3)
    export_onnx.py     →  outputs/model_quantized.onnx          (Stage 4)

To reconstruct the correct quantized graph, this script requires both the
saved checkpoint (model weights) and the quant config (CHOP pass config).
It rebuilds the MaseGraph and re-applies ``quantize_transform_pass`` before
exporting, ensuring the exported graph matches what Stage 3 produced.

The exported ``.onnx`` file can be:
  - Fed into ``onnxsim`` for graph simplification.
  - Compiled with TVM / MASE HLS backend for FPGA/ASIC targets.
  - Evaluated with ``onnxruntime`` for CPU/GPU latency benchmarking.

Usage
-----
    python scripts/export_onnx.py --config  configs/base_config.yaml \\
                                   --quant   configs/quant_params.yaml \\
                                   --checkpoint outputs/checkpoints/checkpoint_quantized.pth \\
                                   --output     outputs/model_quantized.onnx \\
                                   --opset      17
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler import MaseConfigGenerator
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
    parser = argparse.ArgumentParser(description="APQ-Lite — Stage 4: ONNX Export")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--quant", default="configs/quant_params.yaml",
                        help="Quant params YAML — used to reconstruct the CHOP pass config.")
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/checkpoint_quantized.pth",
        help="Quantized checkpoint produced by run_qat.py (Stage 3).",
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
    quant_cfg = load_config(args.quant)

    device = torch.device(args.device)
    logger.info("Export device: %s", device)

    # ---- Build base model and load AdaRounded + quantized weights ----
    num_classes = 10 if cfg["data"]["dataset"] == "cifar10" else 100
    model = build_model(cfg["model"]["architecture"], num_classes=num_classes)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.to(device).eval()
    logger.info("Checkpoint loaded from %s", args.checkpoint)

    # ---- Reconstruct the CHOP quant config and re-apply MASE pass ----
    # The checkpoint holds the weight values but the quantized graph structure
    # (integer operators, scale/zero-point metadata) is only restored by
    # re-applying quantize_transform_pass with the same config used in Stage 3.
    from chop.passes.graph.transforms import quantize_transform_pass
    from chop.passes.graph.analysis import (
        init_metadata_analysis_pass,
        add_common_metadata_analysis_pass,
        add_software_metadata_analysis_pass,
    )
    from chop import MaseGraph

    mk = quant_cfg["mase_keys"]
    generator = MaseConfigGenerator(
        weight_width_key=mk["weight_width_key"],
        activation_width_key=mk["activation_width_key"],
        weight_frac_key=mk["weight_frac_key"],
        activation_frac_key=mk["activation_frac_key"],
        default_frac_width=mk["default_frac_width"],
    )

    # Load the quant_config.json written by run_qat.py rather than
    # regenerating from scratch — ensures bit_map is identical to Stage 3.
    quant_config_path = Path(cfg["project"]["output_dir"]) / "quant_config.json"
    chop_config = generator.load(quant_config_path)
    mase_pass_config = generator.wrap_for_mase_pass(chop_config)
    logger.info("Loaded CHOP quant config from %s", quant_config_path)

    # The analysis passes MUST run before quantize_transform_pass.
    # They populate node.meta["mase"] with op-type, shape, and software
    # metadata that quantize_transform_pass reads on every node.
    dummy_in = {"x": torch.randn(1, 3, 32, 32, device=device)}

    mg = MaseGraph(model)
    mg, _ = init_metadata_analysis_pass(mg)
    mg, _ = add_common_metadata_analysis_pass(mg, pass_args={"dummy_in": dummy_in})
    mg, _ = add_software_metadata_analysis_pass(mg, pass_args={})
    mg, _ = quantize_transform_pass(mg, mase_pass_config)
    model = mg.model
    model.eval()
    logger.info("CHOP quantize_transform_pass re-applied.")

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
