#!/usr/bin/env python3
"""
scripts/run_eval.py
===================
Entry-point for **Evaluation**: Baseline fp32 vs. HA-AdaRound Quantized.

Evaluates two models and writes their results to JSON files consumed by
``notebooks/Results_Visualization.ipynb``:

  outputs/eval_baseline.json   — fp32 model (pre-quantization reference)
  outputs/eval_quantized.json  — AdaRounded + MASE-quantized model

Pipeline position
-----------------
Run this after ``run_qat.py`` (Stage 3) has produced the quantized checkpoint:

    run_profiling.py  →  layer_sensitivity.json          (Stage 1)
    run_qat.py        →  checkpoint_quantized.pth
                         quant_config.json               (Stage 2/2.5/3)
    run_eval.py       →  eval_baseline.json
                         eval_quantized.json             (Evaluation)
    export_onnx.py    →  model_quantized.onnx            (Stage 4)

Usage
-----
    # Evaluate both models (default):
    python scripts/run_eval.py --config  configs/base_config.yaml \\
                                --quant   configs/quant_params.yaml \\
                                --baseline   outputs/checkpoints/checkpoint_best.pth \\
                                --quantized  outputs/checkpoints/checkpoint_quantized.pth

    # Baseline only:
    python scripts/run_eval.py ... --skip-quantized

    # Quantized only:
    python scripts/run_eval.py ... --skip-baseline

    # Include latency benchmark:
    python scripts/run_eval.py ... --benchmark-latency
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.allocator import BitWidthAllocator
from src.compiler import MaseConfigGenerator
from src.engine import Evaluator
from src.models import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger("run_eval")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_val_loader(cfg: dict) -> torch.utils.data.DataLoader:
    """Build the CIFAR validation dataloader."""
    import torchvision
    import torchvision.transforms as T

    val_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset_cls = (
        torchvision.datasets.CIFAR10
        if cfg["data"]["dataset"] == "cifar10"
        else torchvision.datasets.CIFAR100
    )
    val_set = dataset_cls(
        cfg["data"]["data_dir"], train=False, download=True, transform=val_tf
    )
    return torch.utils.data.DataLoader(
        val_set,
        batch_size=cfg["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )


def load_model(cfg: dict, checkpoint: str, device: str) -> nn.Module:
    """Build the model architecture and load weights from checkpoint."""
    num_classes = 10 if cfg["data"]["dataset"] == "cifar10" else 100
    model = build_model(cfg["model"]["architecture"], num_classes=num_classes)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
    return model.to(device).eval()


def apply_mase_quantization(
    model: nn.Module,
    cfg: dict,
    quant_cfg: dict,
    device: str,
) -> nn.Module:
    """Re-apply the CHOP quantize_transform_pass to restore the quantized graph.

    The quantized checkpoint holds AdaRounded weight values but the integer
    operator structure is only restored by re-running the MASE pass with the
    same config used in Stage 3.
    """
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

    quant_config_path = Path(cfg["project"]["output_dir"]) / "quant_config.json"
    chop_config = generator.load(quant_config_path)
    mase_pass_config = generator.wrap_for_mase_pass(chop_config)
    logger.info("Loaded CHOP quant config from %s", quant_config_path)

    dummy_in = {"x": torch.randn(1, 3, 32, 32, device=device)}
    mg = MaseGraph(model)
    mg, _ = init_metadata_analysis_pass(mg)
    mg, _ = add_common_metadata_analysis_pass(mg, pass_args={"dummy_in": dummy_in})
    mg, _ = add_software_metadata_analysis_pass(mg, pass_args={})
    mg, _ = quantize_transform_pass(mg, mase_pass_config)
    logger.info("CHOP quantize_transform_pass applied.")
    return mg.model.eval()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HA-AdaRound — Evaluation: fp32 Baseline vs. Quantized"
    )
    parser.add_argument("--config",  default="configs/base_config.yaml")
    parser.add_argument("--quant",   default="configs/quant_params.yaml")
    parser.add_argument(
        "--baseline",
        default="outputs/checkpoints/checkpoint_best.pth",
        help="fp32 pretrained checkpoint for the baseline evaluation.",
    )
    parser.add_argument(
        "--quantized",
        default="outputs/checkpoints/checkpoint_quantized.pth",
        help="AdaRounded + quantized checkpoint from run_qat.py.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the fp32 baseline evaluation.",
    )
    parser.add_argument(
        "--skip-quantized",
        action="store_true",
        help="Skip the quantized model evaluation.",
    )
    parser.add_argument(
        "--benchmark-latency",
        action="store_true",
        help="Run latency benchmark in addition to accuracy eval.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg      = load_config(args.config)
    quant_cfg = load_config(args.quant)

    torch.manual_seed(cfg["project"]["seed"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    out_dir = Path(cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    val_loader = build_val_loader(cfg)
    loss_fn = nn.CrossEntropyLoss()

    # ---- Evaluate fp32 baseline ----
    if not args.skip_baseline:
        logger.info("--- Baseline (fp32) Evaluation ---")
        baseline_model = load_model(cfg, args.baseline, device)
        evaluator = Evaluator(baseline_model, loss_fn, device=device)

        baseline_results = evaluator.evaluate(val_loader, label="fp32-baseline")

        if args.benchmark_latency:
            latency = evaluator.benchmark_latency(val_loader)
            baseline_results.update(latency)
            logger.info(
                "Baseline latency: %.3f ms/sample",
                latency["latency_ms_per_sample"],
            )

        evaluator.save_results(
            baseline_results, out_dir / "eval_baseline.json", label="fp32-baseline"
        )

    # ---- Evaluate quantized model ----
    if not args.skip_quantized:
        logger.info("--- Quantized (HA-AdaRound) Evaluation ---")
        quant_model = load_model(cfg, args.quantized, device)
        quant_model = apply_mase_quantization(quant_model, cfg, quant_cfg, device)

        evaluator = Evaluator(quant_model, loss_fn, device=device)

        quantized_results = evaluator.evaluate(val_loader, label="ha-adaround")

        if args.benchmark_latency:
            latency = evaluator.benchmark_latency(val_loader)
            quantized_results.update(latency)
            logger.info(
                "Quantized latency: %.3f ms/sample",
                latency["latency_ms_per_sample"],
            )

        evaluator.save_results(
            quantized_results, out_dir / "eval_quantized.json", label="ha-adaround"
        )

    # ---- Model size comparison ----
    # Two complementary views:
    #   1. Checkpoint file size on disk (bytes) — reflects actual storage cost.
    #   2. Theoretical weight memory (param_count × bit_width) — reflects the
    #      compression the allocator targeted regardless of serialisation format.
    logger.info("--- Model Size Comparison ---")

    baseline_path  = Path(args.baseline)
    quantized_path = Path(args.quantized)

    if baseline_path.exists() and quantized_path.exists():
        fp32_bytes  = baseline_path.stat().st_size
        quant_bytes = quantized_path.stat().st_size
        logger.info(
            "  Checkpoint size  fp32=%.2f MB   quantized=%.2f MB   "
            "ratio=%.2fx",
            fp32_bytes / 1e6,
            quant_bytes / 1e6,
            fp32_bytes / max(quant_bytes, 1),
        )

    # Theoretical compression using the allocator's estimate_compression()
    quant_config_path = Path(cfg["project"]["output_dir"]) / "quant_config.json"
    if quant_config_path.exists():
        import json
        with open(quant_config_path) as fh:
            saved_config = json.load(fh)

        # Build bit_map from the saved config (weight_width per layer)
        bit_map = {
            name: {"weight_bits": layer_cfg.get("weight_width", 8), "activation_bits": 8}
            for name, layer_cfg in saved_config.items()
            if isinstance(layer_cfg, dict) and layer_cfg.get("name") == "integer"
        }

        # Count parameters per layer from the baseline model
        baseline_model_ref = load_model(cfg, args.baseline, device="cpu")
        param_counts = {
            name: sum(p.numel() for p in module.parameters())
            for name, module in baseline_model_ref.named_modules()
            if name in bit_map
        }

        allocator = BitWidthAllocator()
        ratio, reduction = allocator.estimate_compression(bit_map, param_counts)
        total_params = sum(param_counts.values())

        logger.info(
            "  Theoretical weight memory  "
            "fp32=%.2f MB   quantized≈%.2f MB   "
            "compression=%.2fx   reduction=%.1f%%",
            (total_params * 32) / 8e6,
            (total_params * 32) / 8e6 / ratio,
            ratio,
            reduction * 100,
        )

    # ---- Print side-by-side accuracy summary if both were run ----
    if not args.skip_baseline and not args.skip_quantized:
        logger.info("--- Accuracy Summary ---")
        for metric in ["top1", "top5", "loss"]:
            if metric in baseline_results and metric in quantized_results:
                delta = quantized_results[metric] - baseline_results[metric]
                logger.info(
                    "  %-6s  baseline=%.4f  quantized=%.4f  delta=%+.4f",
                    metric, baseline_results[metric], quantized_results[metric], delta,
                )


if __name__ == "__main__":
    main()
