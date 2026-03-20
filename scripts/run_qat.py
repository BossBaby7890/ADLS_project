#!/usr/bin/env python3
"""
scripts/run_qat.py
==================
Entry-point for **Stage 2 + 3**: Bit-Width Allocation → QAT Fine-Tuning.

This script reads the sensitivity scores produced by ``run_profiling.py``,
calls the ``BitWidthAllocator`` to assign bit-widths per layer, generates
a MASE/CHOP compatible config via ``MaseConfigGenerator``, and then
fine-tunes the model using Quantization-Aware Training.

Run this from the MASE terminal after Stage 1 has completed.

Usage
-----
    python scripts/run_qat.py --config configs/base_config.yaml \\
                               --quant  configs/quant_params.yaml \\
                               --sensitivity outputs/layer_sensitivity.json \\
                               --pretrained  outputs/checkpoints/checkpoint_best.pth
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

from src.allocator import BitWidthAllocator
from src.compiler import MaseConfigGenerator
from src.engine import Trainer
from src.models import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger("run_qat")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_dataloaders(cfg: dict):
    import torchvision
    import torchvision.transforms as T

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    val_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    dataset_cls = (
        torchvision.datasets.CIFAR10
        if cfg["data"]["dataset"] == "cifar10"
        else torchvision.datasets.CIFAR100
    )
    train_set = dataset_cls(cfg["data"]["data_dir"], train=True, download=True, transform=train_tf)
    val_set = dataset_cls(cfg["data"]["data_dir"], train=False, download=True, transform=val_tf)

    bs = cfg["qat"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return train_loader, val_loader


def build_optimizer_and_scheduler(model: nn.Module, cfg: dict):
    qat_cfg = cfg["qat"]
    optimizer = torch.optim.Adam(model.parameters(), lr=qat_cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=qat_cfg["epochs"] - qat_cfg.get("warmup_epochs", 0),
    )
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="APQ-Lite — Stage 2/3: Allocation + QAT")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--quant", default="configs/quant_params.yaml")
    parser.add_argument(
        "--sensitivity",
        default="outputs/layer_sensitivity.json",
        help="Layer-level sensitivity JSON from run_profiling.py",
    )
    parser.add_argument(
        "--pretrained",
        default=None,
        help="Path to fp32 pretrained checkpoint for QAT warm-start.",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    quant_cfg = load_config(args.quant)

    torch.manual_seed(cfg["project"]["seed"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Load sensitivity scores ----
    with open(args.sensitivity) as fh:
        sensitivity_scores: dict = json.load(fh)
    logger.info("Loaded sensitivity scores for %d layers.", len(sensitivity_scores))

    # ---- Stage 2: Bit-Width Allocation ----
    th = quant_cfg["thresholds"]
    lo = quant_cfg["layer_overrides"]
    allocator = BitWidthAllocator(
        available_bits=quant_cfg["bit_widths"]["available"],
        high_threshold=th["high"],
        mid_threshold=th["mid"],
        first_layer_bits=lo["first_layer_bits"],
        last_layer_bits=lo["last_layer_bits"],
        skip_patterns=lo["skip_layers"],
        weight_default=quant_cfg["bit_widths"]["weight_default"],
        activation_default=quant_cfg["bit_widths"]["activation_default"],
    )
    bit_map = allocator.allocate(sensitivity_scores)

    # ---- Stage 3: MASE/CHOP Config Generation ----
    mk = quant_cfg["mase_keys"]
    generator = MaseConfigGenerator(
        weight_width_key=mk["weight_width_key"],
        activation_width_key=mk["activation_width_key"],
        weight_frac_key=mk["weight_frac_key"],
        activation_frac_key=mk["activation_frac_key"],
        default_frac_width=mk["default_frac_width"],
    )
    chop_config = generator.generate(bit_map)

    out_dir = Path(cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    generator.save(chop_config, out_dir / "quant_config.json")

    mase_pass_config = generator.wrap_for_mase_pass(chop_config)
    logger.info("MASE pass config ready. Keys: %d", len(mase_pass_config))

    # ---- Build model ----
    num_classes = 10 if cfg["data"]["dataset"] == "cifar10" else 100
    model = build_model(cfg["model"]["architecture"], num_classes=num_classes)

    pretrained = args.pretrained or cfg["qat"].get("checkpoint_path")
    if pretrained:
        state = torch.load(pretrained, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
        logger.info("Loaded pretrained weights from %s", pretrained)

    # NOTE: In the MASE environment, replace the block below with:
    #   from chop.passes.graph.transforms import quantize_transform_pass
    #   mg = MaseGraph(model)
    #   mg, _ = quantize_transform_pass(mg, mase_pass_config)
    #   model = mg.model
    logger.info(
        "[MASE env] Insert CHOP quantize_transform_pass here with mase_pass_config."
    )

    # ---- QAT Training ----
    train_loader, val_loader = build_dataloaders(cfg)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        scheduler=scheduler,
        grad_clip=cfg["training"]["grad_clip"],
        checkpoint_dir=cfg["project"]["checkpoint_dir"],
    )

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["qat"]["epochs"],
        save_best=True,
    )

    # Save training history
    hist_path = out_dir / "qat_history.json"
    with open(hist_path, "w") as fh:
        json.dump(history, fh, indent=2)
    logger.info("QAT history saved to %s", hist_path)


if __name__ == "__main__":
    main()
