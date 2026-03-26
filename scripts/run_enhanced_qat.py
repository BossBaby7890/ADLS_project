#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.allocator import BitWidthAllocator
from src.compiler import MaseConfigGenerator
from src.models import build_model
from src.quantization import AdaRoundOptimizer, AdaptiveAdaRoundScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger("run_enhanced_qat")


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

    dataset_cls = (
        torchvision.datasets.CIFAR10
        if cfg["data"]["dataset"] == "cifar10"
        else torchvision.datasets.CIFAR100
    )
    train_set = dataset_cls(
        cfg["data"]["data_dir"],
        train=True,
        download=True,
        transform=train_tf,
    )

    return torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )


def collect_layer_inputs(model, target_layer, calib_inputs, device):
    captured = []

    def _hook(module, inp, out):
        captured.append(inp[0].detach())

    handle = target_layer.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(calib_inputs.to(device))
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(f"No activations captured for layer {target_layer}")
    return captured[0]


def collect_calibration_inputs(calib_loader, calib_batches: int):
    chunks = []
    for i, (imgs, _) in enumerate(calib_loader):
        chunks.append(imgs)
        if i + 1 >= calib_batches:
            break
    if not chunks:
        raise RuntimeError("Calibration loader returned no batches.")
    return torch.cat(chunks, dim=0)


def run_adaround_with_schedule(
    model,
    bit_map,
    sensitivity_scores,
    calib_loader,
    device,
    base_steps,
    base_calib_batches,
):
    scheduler = AdaptiveAdaRoundScheduler(
        base_steps=base_steps,
        min_steps=100,
        max_steps=max(1500, base_steps * 3),
        sensitivity_scale=1.0,
        low_bit_boost=1.5,
        mid_bit_boost=1.0,
        high_bit_boost=0.5,
        base_calib_batches=base_calib_batches,
        max_calib_batches=4,
    )

    schedule = scheduler.build_schedule(bit_map, sensitivity_scores)
    named_modules = dict(model.named_modules())
    model.eval()

    total = sum(1 for n in bit_map if n in named_modules)
    done = 0

    for layer_name, bit_cfg in bit_map.items():
        if layer_name not in named_modules:
            continue

        layer = named_modules[layer_name]
        if not hasattr(layer, "weight") or layer.weight is None:
            continue

        layer_budget = schedule[layer_name]
        n_bits = bit_cfg["weight_bits"]
        n_steps = layer_budget["steps"]
        calib_batches = layer_budget["calib_batches"]

        calib_inputs = collect_calibration_inputs(calib_loader, calib_batches)
        layer_inputs = collect_layer_inputs(model, layer, calib_inputs, device)

        done += 1
        logger.info(
            "Adaptive AdaRound [%d/%d] %s | bits=%d | sens=%.4f | steps=%d | calib_batches=%d",
            done, total, layer_name, n_bits, layer_budget["sensitivity"], n_steps, calib_batches
        )

        opt = AdaRoundOptimizer(n_bits=n_bits)
        opt.optimize_layer(layer, layer_inputs, n_steps=n_steps)
        layer.weight.data = opt.get_rounded_weights()

    logger.info("Adaptive AdaRound complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced APQ-Lite with adaptive AdaRound scheduling")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--quant", default="configs/quant_params.yaml")
    parser.add_argument("--sensitivity", default="outputs/layer_sensitivity.json")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--adaround-steps", type=int, default=500)
    parser.add_argument("--calib-batches", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    quant_cfg = load_config(args.quant)

    torch.manual_seed(cfg["project"]["seed"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    with open(args.sensitivity) as fh:
        sensitivity_scores = json.load(fh)

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

    num_classes = 10 if cfg["data"]["dataset"] == "cifar10" else 100
    model = build_model(cfg["model"]["architecture"], num_classes=num_classes)

    pretrained = args.pretrained or cfg["model"].get("pretrained_path")
    if pretrained:
        state = torch.load(pretrained, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
        logger.info("Loaded pretrained weights from %s", pretrained)

    model = model.to(device)
    train_loader = build_dataloaders(cfg)

    run_adaround_with_schedule(
        model=model,
        bit_map=bit_map,
        sensitivity_scores=sensitivity_scores,
        calib_loader=train_loader,
        device=device,
        base_steps=args.adaround_steps,
        base_calib_batches=args.calib_batches,
    )

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
    generator.save(chop_config, out_dir / "quant_config_enhanced.json")

# Save AdaRounded model first, even if CHOP is unavailable
    ckpt_dir = Path(cfg["project"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    adaround_ckpt_path = ckpt_dir / "checkpoint_adarounded_enhanced.pth"
    torch.save({"model_state_dict": model.state_dict()}, adaround_ckpt_path)
    logger.info("AdaRounded checkpoint saved to %s", adaround_ckpt_path)

# Try CHOP/MASE quantization pass only if available
    try:
        from chop.passes.graph.transforms import quantize_transform_pass
        from chop import MaseGraph

        mg = MaseGraph(model)
        mg, _ = quantize_transform_pass(mg, generator.wrap_for_mase_pass(chop_config))
        model = mg.model

        quant_ckpt_path = ckpt_dir / "checkpoint_quantized_enhanced.pth"
        torch.save({"model_state_dict": model.state_dict()}, quant_ckpt_path)
        logger.info("Enhanced quantized checkpoint saved to %s", quant_ckpt_path)

    except ModuleNotFoundError:
        logger.warning(
            "CHOP/MASE is not installed in this environment. "
            "Saved AdaRounded checkpoint and quant config only. "
            "Run this script inside the MASE environment for full quantize_transform_pass support."
         )


if __name__ == "__main__":
    main()

