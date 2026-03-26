#!/usr/bin/env python3
"""
scripts/run_profiling.py
========================
Entry-point for **Stage 1**: Gradient Sensitivity Profiling.

Run this script from the MASE terminal environment after activating the MASE
conda/venv to accumulate per-layer squared gradient norms and produce a
sensitivity JSON that drives the bit-width allocator.

Usage
-----
    python scripts/run_profiling.py --config configs/base_config.yaml \\
                                    --quant  configs/quant_params.yaml

Outputs
-------
    outputs/sensitivity_scores.json   — raw per-parameter scores
    outputs/layer_sensitivity.json    — scores aggregated to module level
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# Ensure the project root is on sys.path when run from the terminal
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import CalibrationSampleSelector
from src.models import build_model
from src.profiler import GradientSensitivityProfiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger("run_profiling")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_dataset(cfg: dict, split: str = "train"):
    """Build a CIFAR-10/100 dataset from config."""
    import torchvision
    import torchvision.transforms as T

    transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    dataset_cls = (
        torchvision.datasets.CIFAR10
        if cfg["data"]["dataset"] == "cifar10"
        else torchvision.datasets.CIFAR100
    )
    return dataset_cls(
        root=cfg["data"]["data_dir"],
        train=(split == "train"),
        download=True,
        transform=transform,
    )


def build_dataloader_from_dataset(
    dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="APQ-Lite — Stage 1: Sensitivity Profiling")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--quant", default="configs/quant_params.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a pretrained fp32 checkpoint (.pth). "
             "Overrides config value if provided.",
    )
    parser.add_argument("--device", default=None, help="cuda | cpu")
    parser.add_argument(
        "--calib-strategy",
        type=str,
        default="random",
        choices=["random", "class_balanced"],
        help="Calibration subset selection strategy for profiling.",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=256,
        help="Number of samples to use in the profiling calibration subset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    quant_cfg = load_config(args.quant)

    torch.manual_seed(cfg["project"]["seed"])

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    logger.info(
        "Profiling calibration selection | strategy=%s | samples=%d",
        args.calib_strategy,
        args.calib_samples,
    ) 
    # ---- Model ----
    num_classes = 10 if cfg["data"]["dataset"] == "cifar10" else 100
    model = build_model(cfg["model"]["architecture"], num_classes=num_classes)

    checkpoint_path = args.checkpoint or cfg["model"].get("pretrained_path")
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
        logger.info("Loaded weights from %s", checkpoint_path)

    # ---- Dataloader ----
    # ---- Calibration dataset + selector ----
    train_dataset = build_dataset(cfg, split="train")
    selector = CalibrationSampleSelector(
        strategy=args.calib_strategy,
        num_samples=args.calib_samples,
        seed=cfg["project"]["seed"],
    )
    calib_dataset = selector.select(train_dataset)

    dataloader = build_dataloader_from_dataset(
        calib_dataset,
        batch_size=cfg["profiling"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    # ---- Profiler ----
    loss_fn = nn.CrossEntropyLoss()
    profiler = GradientSensitivityProfiler(
        model=model,
        loss_fn=loss_fn,
        device=device,
        normalize=cfg["profiling"]["normalize_scores"],
    )

    param_scores = profiler.profile(
        dataloader=dataloader,
        num_batches=cfg["profiling"]["num_batches"],
    )

    # Save per-parameter scores
    out_dir = Path(cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    profiler.save(cfg["profiling"]["output_file"])

    # Aggregate to module-level and save separately
    layer_scores = GradientSensitivityProfiler.aggregate_to_layers(param_scores, model)
    layer_out = out_dir / "layer_sensitivity.json"
    with open(layer_out, "w") as fh:
        json.dump(layer_scores, fh, indent=2)
    logger.info("Layer-level sensitivity saved to %s", layer_out)

    # Print top-10 most sensitive layers
    sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
    logger.info("Top-10 most sensitive layers:")
    for name, score in sorted_layers[:10]:
        logger.info("  %-45s  %.4f", name, score)


if __name__ == "__main__":
    main()
