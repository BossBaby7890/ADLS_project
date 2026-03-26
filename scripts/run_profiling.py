#!/usr/bin/env python3
"""
scripts/run_profiling.py
========================
Entry-point for **Stage 1**: Joint Sensitivity Profiling (extended).

Collects three metrics in a single calibration pass:
  - Gradient norm per layer       (existing)
  - Hutchinson Hessian trace      (new)
  - Taylor channel scores         (new)

Outputs
-------
    outputs/sensitivity_scores.json   — raw per-parameter gradient norms
    outputs/layer_sensitivity.json    — full per-layer dict with all 3 metrics

Usage
-----
    python scripts/run_profiling.py --config configs/base_config.yaml \\
                                    --quant  configs/quant_params.yaml
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import build_model
from src.profiler.sensitivity import GradientSensitivityProfiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger("run_profiling")


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_dataloader(cfg: dict, split: str = "train") -> torch.utils.data.DataLoader:
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
    dataset = dataset_cls(
        root=cfg["data"]["data_dir"],
        train=(split == "train"),
        download=True,
        transform=transform,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["profiling"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HA-AdaRound — Stage 1: Joint Sensitivity Profiling")
    parser.add_argument("--config",     default="configs/base_config.yaml")
    parser.add_argument("--quant",      default="configs/quant_params.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to pretrained fp32 checkpoint (.pth).")
    parser.add_argument("--device",     default=None, help="cuda | cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg["project"]["seed"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Model ----
    num_classes = 10 if cfg["data"]["dataset"] == "cifar10" else 100
    model = build_model(cfg["model"]["architecture"], num_classes=num_classes)

    checkpoint_path = args.checkpoint or cfg["model"].get("pretrained_path")
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
        logger.info("Loaded weights from %s", checkpoint_path)

    # ---- Dataloader ----
    dataloader = build_dataloader(cfg, split="train")

    # ---- Profiler ----
    hutchinson_probes = cfg["profiling"].get("hutchinson_probes", 3)
    profiler = GradientSensitivityProfiler(
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
        normalize=cfg["profiling"]["normalize_scores"],
        hutchinson_probes=hutchinson_probes,
    )

    param_scores = profiler.profile(
        dataloader=dataloader,
        num_batches=cfg["profiling"]["num_batches"],
    )

    out_dir = Path(cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw per-parameter gradient norms (backward compat)
    profiler.save(cfg["profiling"]["output_file"])

    # Save full rich layer-level dict (grad_norm + hessian_trace + taylor_scores)
    # This is what run_search.py and the search engine consume
    layer_out = cfg["profiling"].get("layer_output_file", "outputs/layer_sensitivity.json")
    profiler.save_full(layer_out, model)

    # Print top-10 most sensitive layers by grad_norm
    full_scores = profiler.get_full_layer_scores(model)
    sorted_layers = sorted(
        full_scores.items(),
        key=lambda x: x[1].get("grad_norm", 0.0),
        reverse=True,
    )
    logger.info("Top-10 most sensitive layers (grad_norm):")
    for name, scores in sorted_layers[:10]:
        logger.info(
            "  %-45s  grad_norm=%.4f  hessian_trace=%.4f  n_channels=%d",
            name,
            scores.get("grad_norm", 0.0),
            scores.get("hessian_trace", 0.0),
            len(scores.get("taylor_scores", [])),
        )


if __name__ == "__main__":
    main()
