#!/usr/bin/env python3
"""
debug_bn.py — Temporary diagnostic script (DO NOT COMMIT).

Loads the fp32 checkpoint and the AdaRound quantized checkpoint.
For each model:
  - Prints running_mean, running_var, weight (gamma), bias (beta) for the
    first three BatchNorm layers.
  - Runs a single forward pass through the first Conv2d layer on a fixed
    random input and prints output min, max, mean abs.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models import build_model

FP32_CKPT  = ROOT / "outputs/checkpoints/checkpoint_best.pth"
QUANT_CKPT = ROOT / "outputs/checkpoints/checkpoint_quantized.pth"
ARCH       = "resnet20"
NUM_CLASSES = 10

torch.manual_seed(0)
FIXED_INPUT = torch.randn(1, 3, 32, 32)   # same tensor for both models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(ckpt_path: Path) -> nn.Module:
    model = build_model(ARCH, num_classes=NUM_CLASSES)
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def print_bn_stats(model: nn.Module, label: str, n: int = 3) -> None:
    print(f"\n{'='*60}")
    print(f"  BatchNorm stats — {label}")
    print(f"{'='*60}")
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        print(f"\n  Layer : {name}")
        print(f"    running_mean  : {module.running_mean[:8].numpy().round(6).tolist()}")
        print(f"    running_var   : {module.running_var[:8].numpy().round(6).tolist()}")
        if module.weight is not None:
            print(f"    gamma (weight): {module.weight.data[:8].detach().numpy().round(6).tolist()}")
        if module.bias is not None:
            print(f"    beta  (bias)  : {module.bias.data[:8].detach().numpy().round(6).tolist()}")
        count += 1
        if count >= n:
            break


def print_first_conv_output(model: nn.Module, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  First Conv2d forward pass — {label}")
    print(f"{'='*60}")
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        with torch.no_grad():
            out = module(FIXED_INPUT)
        print(f"  Layer      : {name}  {tuple(module.weight.shape)}")
        print(f"  Output shape: {tuple(out.shape)}")
        print(f"  min        : {out.min().item():.6f}")
        print(f"  max        : {out.max().item():.6f}")
        print(f"  mean |out| : {out.abs().mean().item():.6f}")
        break   # first conv only


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for path, label in [(FP32_CKPT, "fp32 baseline"), (QUANT_CKPT, "AdaRound quantized")]:
        if not path.exists():
            print(f"[ERROR] Checkpoint not found: {path}")
            continue

        print(f"\n{'#'*60}")
        print(f"  Loading: {path.name}")
        print(f"{'#'*60}")

        model = load(path)
        print_bn_stats(model, label)
        print_first_conv_output(model, label)


if __name__ == "__main__":
    main()
