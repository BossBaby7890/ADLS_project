#!/usr/bin/env python3
"""
scripts/run_qat.py
==================
Entry-point for **Stage 2 + 2.5 + 3**: Bit-Width Allocation → AdaRound → MASE Config.

This script reads the sensitivity scores produced by ``run_profiling.py``,
calls the ``BitWidthAllocator`` to assign bit-widths per layer (Stage 2),
optionally runs AdaRound learned rounding on each quantizable layer using a
small calibration batch (Stage 2.5), and generates a MASE/CHOP compatible
config via ``MaseConfigGenerator`` (Stage 3).

Run this from the MASE terminal after Stage 1 has completed.

Usage
-----
    # With AdaRound (default):
    python scripts/run_qat.py --config configs/base_config.yaml \\
                               --quant  configs/quant_params.yaml \\
                               --sensitivity outputs/layer_sensitivity.json \\
                               --pretrained  outputs/checkpoints/checkpoint_best.pth

    # Skip AdaRound (RTN only, faster):
    python scripts/run_qat.py ... --skip-adaround

    # Tune AdaRound:
    python scripts/run_qat.py ... --adaround-steps 1000 --calib-batches 4
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
from src.models import build_model
from src.quantization import AdaRoundOptimizer
# QAT removed — AdaRound handles post-quantization error correction

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

    bs = cfg["data"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    # val_loader removed — QAT removed; AdaRound calibration uses train_loader only
    return train_loader


def collect_layer_inputs(
    model: nn.Module,
    target_layer: nn.Module,
    calib_inputs: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Capture the input activations arriving at ``target_layer`` via a hook.

    Registers a temporary forward hook on ``target_layer``, runs one forward
    pass of the full model with ``calib_inputs``, records ``input[0]`` (the
    activation tensor that multiplies the layer weight), then removes the hook.

    This gives AdaRound exactly the calibration data it needs: the real
    distribution of activations seen by that layer during inference.

    Parameters
    ----------
    model:
        The full model (needed to run the forward pass up to ``target_layer``).
    target_layer:
        The specific ``nn.Conv2d`` or ``nn.Linear`` being AdaRounded.
    calib_inputs:
        A batch of raw model inputs (images), already on CPU — moved to
        ``device`` inside this function.
    device:
        Device string matching the model's device.

    Returns
    -------
    torch.Tensor
        The captured activation batch, detached from the graph.
    """
    captured: list[torch.Tensor] = []

    def _hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        # inp is a tuple; inp[0] is the activation fed into this layer
        captured.append(inp[0].detach())

    handle = target_layer.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(calib_inputs.to(device))
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(
            f"Hook on {type(target_layer).__name__} captured nothing — "
            "did the calibration batch reach that layer?"
        )
    return captured[0]


def run_adaround(
    model: nn.Module,
    bit_map: dict,
    calib_loader: torch.utils.data.DataLoader,
    device: str,
    n_steps: int,
    calib_batches: int,
) -> None:
    """Stage 2.5: run AdaRound in-place on every quantizable layer.

    For each layer present in ``bit_map`` (i.e. every layer the allocator
    assigned a bit-width to), this function:

      1. Collects input activations by running calibration data through the
         model up to that layer.
      2. Runs ``AdaRoundOptimizer.optimize_layer()`` to learn the optimal
         rounding for that layer's weights.
      3. Snaps the learned weights back via ``layer.weight.data = ...``.

    This mutates ``model`` in-place; no return value.

    Parameters
    ----------
    model:
        The fp32 model (pretrained weights already loaded).
    bit_map:
        Output of ``BitWidthAllocator.allocate()`` — determines which layers
        to AdaRound and at what bit-width.
    calib_loader:
        DataLoader used to draw calibration batches.
    device:
        Device string.
    n_steps:
        AdaRound optimisation steps per layer (paper default: 10 000,
        lightweight default: 500).
    calib_batches:
        Number of batches to concatenate for calibration.  More batches →
        better activation statistics, but slower hook collection.
    """
    # --- Collect calibration inputs (a few batches concatenated) -----------
    calib_chunks: list[torch.Tensor] = []
    for i, (imgs, _) in enumerate(calib_loader):
        calib_chunks.append(imgs)
        if i + 1 >= calib_batches:
            break
    calib_inputs = torch.cat(calib_chunks, dim=0)  # (N, C, H, W)
    logger.info(
        "AdaRound calibration set: %d images from %d batch(es).",
        calib_inputs.shape[0], calib_batches,
    )

    # Build a name → module lookup for the layers we need to AdaRound
    named_modules = dict(model.named_modules())

    model.eval()  # no dropout / BN update during calibration
    total = sum(1 for n in bit_map if n in named_modules)
    done = 0

    for layer_name, bit_cfg in bit_map.items():
        if layer_name not in named_modules:
            logger.debug("AdaRound: skipping '%s' (not found in model).", layer_name)
            continue

        layer = named_modules[layer_name]

        if not hasattr(layer, "weight") or layer.weight is None:
            logger.debug("AdaRound: skipping '%s' (no weight tensor).", layer_name)
            continue

        n_bits = bit_cfg["weight_bits"]
        done += 1
        logger.info(
            "AdaRound [%d/%d] %-40s  %d-bit  steps=%d",
            done, total, layer_name, n_bits, n_steps,
        )

        # Capture activations reaching this specific layer
        layer_inputs = collect_layer_inputs(model, layer, calib_inputs, device)

        # Optimise rounding variables V for this layer
        opt = AdaRoundOptimizer(n_bits=n_bits)
        opt.optimize_layer(layer, layer_inputs, n_steps=n_steps)

        # Snap learned weights back into the model graph
        layer.weight.data = opt.get_rounded_weights()

    logger.info("AdaRound complete — %d/%d layers rounded.", done, len(bit_map))


# QAT removed — build_optimizer_and_scheduler() removed with it

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="APQ-Lite — Stage 2/2.5/3: Allocation + AdaRound + MASE Config")
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
        help="Path to fp32 pretrained checkpoint to load before AdaRound.",
    )
    parser.add_argument("--device", default=None)

    # AdaRound controls
    parser.add_argument(
        "--skip-adaround",
        action="store_true",
        help="Skip Stage 2.5 and use standard RTN rounding (faster, lower accuracy).",
    )
    parser.add_argument(
        "--adaround-steps",
        type=int,
        default=500,
        help="AdaRound optimisation steps per layer (default: 500).",
    )
    parser.add_argument(
        "--calib-batches",
        type=int,
        default=1,
        help="Number of calibration batches to collect for AdaRound (default: 1).",
    )

    # Debug / sanity-check mode
    parser.add_argument(
        "--force-bits",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Bypass sensitivity profiling and assign N bits to every layer. "
            "Skips the allocator and ignores --sensitivity. "
            "Example: --force-bits 8  (useful for debugging accuracy.)"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    quant_cfg = load_config(args.quant)

    torch.manual_seed(cfg["project"]["seed"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Build model (needed before AdaRound) ----
    num_classes = 10 if cfg["data"]["dataset"] == "cifar10" else 100
    model = build_model(cfg["model"]["architecture"], num_classes=num_classes)

    pretrained = args.pretrained or cfg["model"].get("pretrained_path")
    if pretrained:
        state = torch.load(pretrained, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
        logger.info("Loaded pretrained weights from %s", pretrained)

    model = model.to(device)

    # ---- Stage 2: Bit-Width Allocation ----
    if args.force_bits is not None:
        logger.info(
            "DEBUG --force-bits %d: bypassing allocator, assigning %d-bit to all "
            "weight-bearing layers (sensitivity file ignored).",
            args.force_bits, args.force_bits,
        )
        bit_map = {
            name: {"weight_bits": args.force_bits, "activation_bits": args.force_bits}
            for name, module in model.named_modules()
            if hasattr(module, "weight") and module.weight is not None
        }
        logger.info("force-bits bit_map: %d layers assigned %d-bit.", len(bit_map), args.force_bits)
    else:
        # Load sensitivity scores produced by run_profiling.py
        with open(args.sensitivity) as fh:
            sensitivity_scores: dict = json.load(fh)
        logger.info("Loaded sensitivity scores for %d layers.", len(sensitivity_scores))

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

    # ---- Build dataloader (needed for AdaRound calibration) ----
    train_loader = build_dataloaders(cfg)

    # ---- Stage 2.5: AdaRound — Learned Rounding ----
    if args.skip_adaround:
        logger.info("Stage 2.5: AdaRound skipped (--skip-adaround). Using RTN.")
    else:
        logger.info(
            "Stage 2.5: AdaRound  steps=%d  calib_batches=%d",
            args.adaround_steps, args.calib_batches,
        )
        run_adaround(
            model=model,
            bit_map=bit_map,
            calib_loader=train_loader,
            device=device,
            n_steps=args.adaround_steps,
            calib_batches=args.calib_batches,
        )

    # ---- Stage 3: MASE/CHOP Config Generation ----
    # By this point model weights already reflect AdaRound rounding decisions.
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

    # ---- Stage 3 (cont.): Apply CHOP quantization transform pass ----
    # Converts the AdaRounded fp32 model into a quantized MASE graph where
    # each layer is replaced with its integer-arithmetic equivalent according
    # to mase_pass_config.
    #
    # The two analysis passes MUST run before quantize_transform_pass.
    # They populate node.meta["mase"] with op-type, shape, and software
    # metadata that quantize_transform_pass reads on every node.
    # Skipping them causes KeyError: 'mase'.
    from chop.passes.graph.transforms import quantize_transform_pass
    from chop.passes.graph.analysis import (
        init_metadata_analysis_pass,
        add_common_metadata_analysis_pass,
        add_software_metadata_analysis_pass,
    )
    from chop import MaseGraph

    # Dummy input used by add_common_metadata_analysis_pass to trace shapes
    dummy_input = {"x": torch.randn(1, 3, 32, 32, device=device)}

    mg = MaseGraph(model)
    mg, _ = init_metadata_analysis_pass(mg)
    mg, _ = add_common_metadata_analysis_pass(mg, pass_args={"dummy_in": dummy_input})
    mg, _ = add_software_metadata_analysis_pass(mg, pass_args={})
    mg, _ = quantize_transform_pass(mg, mase_pass_config)
    model = mg.model
    logger.info("CHOP quantize_transform_pass applied.")

    # ---- Save quantized checkpoint for export ----
    # Writes the state dict of the MASE-quantized model so that
    # export_onnx.py can load it and re-apply the CHOP pass before exporting.
    ckpt_dir = Path(cfg["project"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "checkpoint_quantized.pth"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    logger.info("Quantized checkpoint saved → %s", ckpt_path)


if __name__ == "__main__":
    main()
