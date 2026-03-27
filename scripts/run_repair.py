

#!/usr/bin/env python3
"""
scripts/run_repair.py
=====================
Post-AdaRound selective precision repair stage.

This script loads the already-quantized checkpoint + quant_config, then
greedily upgrades a few layers (2->4 or 4->8) if doing so gives the best
LOSS reduction per added bit-cost under a repair budget.

Outputs:
- outputs/quant_config_repaired.json
- outputs/eval_repaired.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler import MaseConfigGenerator
from src.engine import Evaluator
from src.models import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
)
logger = logging.getLogger("run_repair")


def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_val_loader(cfg: dict, max_samples: int | None = None):
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Subset

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

    if max_samples is not None:
        val_set = Subset(val_set, list(range(min(max_samples, len(val_set)))))

    return DataLoader(
        val_set,
        batch_size=cfg["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )


def load_model(cfg: dict, checkpoint: str, device: str) -> nn.Module:
    num_classes = 10 if cfg["data"]["dataset"] == "cifar10" else 100
    model = build_model(cfg["model"]["architecture"], num_classes=num_classes)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
    return model.to(device).eval()


def apply_mase_quantization(model: nn.Module, chop_config: dict, device: str) -> nn.Module:
    from chop.passes.graph.transforms import quantize_transform_pass
    from chop.passes.graph.analysis import (
        init_metadata_analysis_pass,
        add_common_metadata_analysis_pass,
        add_software_metadata_analysis_pass,
    )
    from chop import MaseGraph

    generator = MaseConfigGenerator()
    mase_pass_config = generator.wrap_for_mase_pass(chop_config)

    dummy_in = {"x": torch.randn(1, 3, 32, 32, device=device)}
    mg = MaseGraph(model)
    mg, _ = init_metadata_analysis_pass(mg)
    mg, _ = add_common_metadata_analysis_pass(mg, pass_args={"dummy_in": dummy_in})
    mg, _ = add_software_metadata_analysis_pass(mg, pass_args={})
    mg, _ = quantize_transform_pass(mg, mase_pass_config)
    return mg.model.eval()


def evaluate_quantized_model(cfg: dict, quant_cfg: dict, checkpoint: str, device: str, max_samples: int):
    val_loader = build_val_loader(cfg, max_samples=max_samples)
    loss_fn = nn.CrossEntropyLoss()

    model = load_model(cfg, checkpoint, device)
    model = apply_mase_quantization(model, quant_cfg, device)

    evaluator = Evaluator(model, loss_fn, device=device, topk=(1,))
    results = evaluator.evaluate(val_loader, label="repair-eval")
    return results


def load_sensitivity_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def get_layer_param_counts(model: nn.Module) -> dict:
    counts = {}
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            counts[name] = int(module.weight.numel())
    return counts


def step_upgrade_bits(layer_cfg: dict) -> dict | None:
    """
    One-step upgrade only:
      2 -> 4
      4 -> 8
      8 -> None
    """
    current = layer_cfg["weight_width"]
    if current == 2:
        new_bits = 4
    elif current == 4:
        new_bits = 8
    else:
        return None

    new_cfg = copy.deepcopy(layer_cfg)
    new_cfg["weight_width"] = new_bits
    new_cfg["activation_width"] = (
        new_bits if "activation_width" in new_cfg else new_cfg.get("data_in_width", new_bits)
    )
    if "data_in_width" in new_cfg:
        new_cfg["data_in_width"] = new_bits
    if "bias_width" in new_cfg:
        new_cfg["bias_width"] = new_bits

    frac = max(new_bits - 2, 1)
    if "weight_frac_width" in new_cfg:
        new_cfg["weight_frac_width"] = frac
    if "data_in_frac_width" in new_cfg:
        new_cfg["data_in_frac_width"] = frac
    if "bias_frac_width" in new_cfg:
        new_cfg["bias_frac_width"] = frac

    return new_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Post-AdaRound selective precision repair")
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--quant", default="configs/quant_params.yaml")
    parser.add_argument("--quant-config", default="outputs/quant_config.json")
    parser.add_argument("--sensitivity", default="outputs/layer_sensitivity.json")
    parser.add_argument("--quantized", default="outputs/checkpoints/checkpoint_quantized.pth")
    parser.add_argument("--device", default=None)

    parser.add_argument(
        "--repair-budget-frac",
        type=float,
        default=0.10,
        help="Allowed extra bit-cost as fraction of current quantized bit-cost",
    )
    parser.add_argument(
        "--topk-candidates",
        type=int,
        default=12,
        help="Only consider top-k sensitive currently-upgradable layers",
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=5,
        help="Maximum number of repair upgrades to apply",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=2000,
        help="Validation subset size for fast repair scoring",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    _ = load_config(args.quant)  # kept for symmetry / future use
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Device: %s", device)

    quant_config_path = Path(args.quant_config)
    with open(quant_config_path) as fh:
        chop_config = json.load(fh)

    sensitivities = load_sensitivity_json(args.sensitivity)

    # Build model once just to get layer sizes
    model_for_counts = load_model(cfg, args.quantized, device)
    param_counts = get_layer_param_counts(model_for_counts)

    # Baseline current quantized score
    current_results = evaluate_quantized_model(
        cfg, chop_config, args.quantized, device, max_samples=args.eval_samples
    )
    current_top1 = current_results["top1"]
    current_loss = current_results["loss"]

    logger.info(
        "Initial quantized metrics: top1 = %.4f | loss = %.4f",
        current_top1,
        current_loss,
    )

    def total_bit_cost(config):
        total = 0
        for layer_name, layer_cfg in config.items():
            n = param_counts.get(layer_name, 0)
            total += n * layer_cfg["weight_width"]
        return total

    current_cost = total_bit_cost(chop_config)
    repair_budget = current_cost * args.repair_budget_frac
    used_budget = 0.0

    candidates = []
    for layer_name, layer_cfg in chop_config.items():
        if layer_name not in sensitivities:
            continue
        if layer_cfg["weight_width"] not in (2, 4):
            continue
        candidates.append((layer_name, sensitivities[layer_name]))

    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = candidates[: args.topk_candidates]

    logger.info("Candidate layers considered for repair: %d", len(candidates))

    applied_repairs = []

    for repair_step in range(args.max_repairs):
        best_layer = None
        best_score = 0.0
        best_results = None
        best_new_cfg = None
        best_extra_cost = None

        for layer_name, sens in candidates:
            if any(r["layer"] == layer_name for r in applied_repairs):
                continue

            upgraded = step_upgrade_bits(chop_config[layer_name])
            if upgraded is None:
                continue

            extra_cost = param_counts.get(layer_name, 0) * (
                upgraded["weight_width"] - chop_config[layer_name]["weight_width"]
            )

            if used_budget + extra_cost > repair_budget:
                continue

            trial_config = copy.deepcopy(chop_config)
            trial_config[layer_name] = upgraded

            trial_results = evaluate_quantized_model(
                cfg, trial_config, args.quantized, device, max_samples=args.eval_samples
            )

            trial_top1 = trial_results["top1"]
            trial_loss = trial_results["loss"]

            delta_loss = current_loss - trial_loss  # positive = improvement
            score = delta_loss / max(extra_cost, 1)

            logger.info(
                "Repair candidate %-40s sens=%.4f Δloss=%.6f Δtop1=%.4f extra_cost=%d score=%.6e",
                layer_name,
                sens,
                delta_loss,
                trial_top1 - current_top1,
                int(extra_cost),
                score,
            )

            if score > best_score:
                best_score = score
                best_layer = layer_name
                best_results = trial_results
                best_new_cfg = upgraded
                best_extra_cost = extra_cost

        if best_layer is None or best_score <= 0:
            logger.info("No beneficial repair found at step %d. Stopping.", repair_step + 1)
            break

        chop_config[best_layer] = best_new_cfg
        used_budget += best_extra_cost
        current_results = best_results
        current_top1 = current_results["top1"]
        current_loss = current_results["loss"]

        applied_repairs.append({
            "layer": best_layer,
            "new_bits": best_new_cfg["weight_width"],
            "extra_cost": int(best_extra_cost),
            "top1_after": current_top1,
            "loss_after": current_loss,
        })

        logger.info(
            "Applied repair %d: %s -> %d-bit | used_budget=%.0f / %.0f | top1=%.4f | loss=%.4f",
            repair_step + 1,
            best_layer,
            best_new_cfg["weight_width"],
            used_budget,
            repair_budget,
            current_top1,
            current_loss,
        )

    out_dir = Path(cfg["project"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    repaired_config_path = out_dir / "quant_config_repaired.json"
    with open(repaired_config_path, "w") as fh:
        json.dump(chop_config, fh, indent=2)

    repaired_eval_path = out_dir / "eval_repaired.json"
    payload = {
        "metrics": current_results,
        "repair_budget_frac": args.repair_budget_frac,
        "used_budget": used_budget,
        "applied_repairs": applied_repairs,
    }
    with open(repaired_eval_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Saved repaired config to %s", repaired_config_path)
    logger.info("Saved repaired metrics to %s", repaired_eval_path)


if __name__ == "__main__":
    main()
