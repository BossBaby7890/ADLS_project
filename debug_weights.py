#!/usr/bin/env python3
"""
debug_weights.py — Temporary diagnostic script (DO NOT COMMIT).

Loads the quantized checkpoint and inspects weight tensors to determine
whether stored values are raw integers or dequantized floats.
"""

import sys
from pathlib import Path

import torch

CKPT_PATH = Path("outputs/checkpoints/checkpoint_quantized.pth")

def classify_range(w_min: float, w_max: float, w_mean_abs: float) -> str:
    abs_max = max(abs(w_min), abs(w_max))
    if abs_max > 10.0:
        return "RAW INTEGERS  (range suggests stored as integer-valued floats, e.g. -128..127)"
    elif abs_max <= 2.0:
        return "DEQUANTIZED FLOATS  (range is consistent with fp32 weights or scaled output)"
    else:
        return f"AMBIGUOUS  (abs_max={abs_max:.4f} — neither clearly integer nor unit-range float)"

def main():
    if not CKPT_PATH.exists():
        print(f"[ERROR] Checkpoint not found: {CKPT_PATH}")
        sys.exit(1)

    state = torch.load(CKPT_PATH, map_location="cpu")
    # Support both bare state_dict and wrapped {"model_state_dict": ...}
    sd = state.get("model_state_dict", state)

    print(f"Checkpoint : {CKPT_PATH}")
    print(f"Keys in file: {list(state.keys()) if isinstance(state, dict) else type(state)}")
    print(f"Total tensors in state_dict: {len(sd)}\n")

    printed = 0
    for key, tensor in sd.items():
        if tensor.ndim < 1:
            continue
        # Skip non-weight tensors (bias, running_mean, etc.)
        if not any(tag in key for tag in ("weight",)):
            continue
        if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            print(f"  {key}: dtype={tensor.dtype}  shape={tuple(tensor.shape)}  — skipped (not float)")
            continue

        w = tensor.float()
        w_min      = w.min().item()
        w_max      = w.max().item()
        w_mean_abs = w.abs().mean().item()
        verdict    = classify_range(w_min, w_max, w_mean_abs)

        print(f"Layer : {key}")
        print(f"  shape      : {tuple(tensor.shape)}")
        print(f"  dtype      : {tensor.dtype}")
        print(f"  min        : {w_min:.6f}")
        print(f"  max        : {w_max:.6f}")
        print(f"  mean |w|   : {w_mean_abs:.6f}")
        print(f"  verdict    : {verdict}")
        print()

        printed += 1
        if printed >= 5:
            print(f"... (showing first {printed} weight tensors only)")
            break

    if printed == 0:
        print("[WARNING] No float weight tensors found in checkpoint.")

if __name__ == "__main__":
    main()
