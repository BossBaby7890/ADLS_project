"""
scripts/run_search.py
=====================
Master entry-point for the expanded HA-AdaRound pipeline.

Orchestrates Stages 1.5 through 3.5:

  Stage 1.5  — Policy search (greedy + CMA-ES in parallel, best wins)
  Stage 2    — Taylor structured pruning
  Stage 2.2  — KD recovery fine-tuning
  Stage 2.5  — AdaRound (existing, applied to pruned model)
  Stage 3    — MASE compiler (existing)
  Stage 3.5  — Resource parser + cost model recalibration

Prerequisites (must be run first):
  python scripts/run_profiling.py ...   →  outputs/layer_sensitivity.json
  python scripts/run_qat.py ...         →  outputs/checkpoints/checkpoint_best.pth

Usage
-----
python scripts/run_search.py \\
    --config     configs/base_config.yaml \\
    --quant      configs/quant_params.yaml \\
    --sensitivity outputs/layer_sensitivity.json \\
    --pretrained  outputs/checkpoints/checkpoint_best.pth \\
    --bops-budget 0.35 \\
    --max-sparsity 0.5 \\
    --pruning-rounds 4 \\
    --kd-epochs 5 \\
    --search-strategy both \\
    --hardware-target fpga

Flags
-----
--search-strategy  both | greedy | cmaes
    ``both``   runs greedy and CMA-ES in parallel, takes the better result.
    ``greedy`` runs only the greedy search (faster, good baseline).
    ``cmaes``  runs only CMA-ES (slower, more thorough).

--skip-proxy-collection
    Skip generating new proxy training data.  Requires
    ``outputs/proxy_data.pt`` to already exist from a previous run.

--skip-adaround
    Skip AdaRound and use RTN rounding only (much faster, lower accuracy).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without install
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import build_model
from src.hardware.cost_model import CostModel
from src.search.accuracy_proxy import AccuracyProxy, ProxyDataCollector
from src.search.greedy_search import GreedySearch
from src.search.cmaes_search import CMAESSearch
from src.pruning.taylor_pruner import TaylorPruner
from src.pruning.kd_recovery import KDRecovery
from src.quantization.adaround import AdaRoundOptimizer
from src.compiler.mase_integration import MaseConfigGenerator
from src.compiler.resource_parser import ResourceParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_search")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HA-AdaRound policy search pipeline")

    p.add_argument("--config",      required=True, help="Path to base_config.yaml")
    p.add_argument("--quant",       required=True, help="Path to quant_params.yaml")
    p.add_argument("--sensitivity", required=True, help="Path to layer_sensitivity.json")
    p.add_argument("--pretrained",  required=True, help="Path to FP32 checkpoint .pth")

    p.add_argument("--bops-budget",    type=float, default=0.35,
                   help="Max BOPs as fraction of fp32 baseline (default 0.35)")
    p.add_argument("--max-sparsity",   type=float, default=0.50,
                   help="Max per-layer pruning ratio (default 0.50)")
    p.add_argument("--pruning-rounds", type=int,   default=4,
                   help="Iterative pruning rounds (default 4)")
    p.add_argument("--kd-epochs",      type=int,   default=5,
                   help="KD recovery epochs per pruning round (default 5)")
    p.add_argument("--hardware-target", default="fpga",
                   choices=["cpu", "gpu", "fpga"],
                   help="Hardware target for cost model (default fpga)")

    p.add_argument("--search-strategy", default="both",
                   choices=["both", "greedy", "cmaes"],
                   help="Which search(es) to run (default both)")
    p.add_argument("--cmaes-generations", type=int, default=150,
                   help="Max CMA-ES generations (default 150)")
    p.add_argument("--cmaes-popsize",     type=int, default=20,
                   help="CMA-ES population size (default 20)")
    p.add_argument("--proxy-policies",    type=int, default=150,
                   help="Random policies for proxy training data (default 150)")

    p.add_argument("--skip-proxy-collection", action="store_true",
                   help="Reuse cached proxy training data from outputs/proxy_data.pt")
    p.add_argument("--skip-adaround",         action="store_true",
                   help="Use RTN rounding instead of AdaRound (faster)")
    p.add_argument("--adaround-steps", type=int, default=500,
                   help="AdaRound optimisation steps per layer (default 500)")
    p.add_argument("--calib-batches",  type=int, default=1,
                   help="Calibration batches for AdaRound (default 1)")

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_sensitivity(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def get_cifar_loaders(cfg: dict):
    """Build CIFAR-10/100 train and val DataLoaders from base_config."""
    import torchvision
    import torchvision.transforms as T

    dataset_name = cfg["data"]["dataset"]
    data_dir = cfg["data"]["data_dir"]
    batch_size = cfg["data"].get("batch_size", 128)
    num_workers = cfg["data"].get("num_workers", 4)

    norm_mean = (0.4914, 0.4822, 0.4465)
    norm_std  = (0.2023, 0.1994, 0.2010)

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(norm_mean, norm_std),
    ])
    transform_val = T.Compose([
        T.ToTensor(),
        T.Normalize(norm_mean, norm_std),
    ])

    DatasetCls = (
        torchvision.datasets.CIFAR100
        if dataset_name == "cifar100"
        else torchvision.datasets.CIFAR10
    )

    train_ds = DatasetCls(data_dir, train=True,  download=True, transform=transform_train)
    val_ds   = DatasetCls(data_dir, train=False, download=True, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["evaluation"].get("batch_size", 256),
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def select_best_policy(
    policies: dict,
    layer_names: list,
    sensitivity: dict,
    proxy: AccuracyProxy,
    cost_model: CostModel,
) -> tuple:
    """Return (best_policy, strategy_name) from a dict of named policies."""
    best_name = None
    best_score = -float("inf")
    best_policy = None

    from src.search.accuracy_proxy import encode_policy_features

    for name, policy in policies.items():
        features = encode_policy_features(layer_names, sensitivity, policy)
        acc = proxy.predict(features)
        bops_r = cost_model.bops_ratio(policy)
        logger.info("  [%s] predicted_acc=%.4f  bops_ratio=%.3f", name, acc, bops_r)
        if acc > best_score:
            best_score = acc
            best_name = name
            best_policy = policy

    logger.info("Best search strategy: %s (predicted_acc=%.4f)", best_name, best_score)
    return best_policy, best_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg   = load_yaml(args.config)
    qcfg  = load_yaml(args.quant)
    sensitivity = load_sensitivity(args.sensitivity)

    output_dir = Path(cfg["project"]["output_dir"])
    ckpt_dir   = Path(cfg["project"]["checkpoint_dir"])
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------ #
    # 1. Load FP32 model                                                   #
    # ------------------------------------------------------------------ #
    logger.info("Loading FP32 model: %s", cfg["model"]["architecture"])
    num_classes = 100 if cfg["data"]["dataset"] == "cifar100" else 10
    model = build_model(cfg["model"]["architecture"], num_classes=num_classes)
    ckpt = torch.load(args.pretrained, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # Keep a frozen copy as the KD teacher
    teacher = copy.deepcopy(model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------ #
    # 2. Data loaders                                                       #
    # ------------------------------------------------------------------ #
    train_loader, val_loader = get_cifar_loaders(cfg)

    # ------------------------------------------------------------------ #
    # 3. Build cost model                                                   #
    # ------------------------------------------------------------------ #
    logger.info("Building cost model (target: %s)", args.hardware_target)
    cost_model = CostModel(hardware_target=args.hardware_target)
    cost_model.build_from_model(model, input_size=(1, 3, 32, 32), device=device)
    layer_names = cost_model.get_all_layer_names()
    logger.info("Cost model covers %d layers", len(layer_names))

    # ------------------------------------------------------------------ #
    # 4. Train accuracy proxy                                               #
    # ------------------------------------------------------------------ #
    n_features = len(layer_names) * 6
    proxy = AccuracyProxy(n_features=n_features).to(device)
    proxy_cache = output_dir / "proxy_data.pt"
    proxy_weights = output_dir / "proxy_weights.pt"

    if proxy_weights.exists() and args.skip_proxy_collection:
        logger.info("Loading cached proxy weights from %s", proxy_weights)
        proxy.load(proxy_weights)
    else:
        collector = ProxyDataCollector(
            model=model,
            layer_names=layer_names,
            sensitivity=sensitivity,
            device=device,
        )
        proxy_dataset = collector.collect(
            val_loader,
            n_policies=args.proxy_policies,
            eval_batches=5,
            cache_path=proxy_cache if not args.skip_proxy_collection else None,
        )
        proxy.fit(proxy_dataset, epochs=50)
        proxy.save(proxy_weights)

    proxy = proxy.to(device)

    # ------------------------------------------------------------------ #
    # 5. Policy search (greedy + CMA-ES, best wins)                         #
    # ------------------------------------------------------------------ #
    skip_patterns = qcfg["layer_overrides"].get("skip_layers", ["bn", "shortcut"])
    candidate_policies: dict = {}

    if args.search_strategy in ("both", "greedy"):
        logger.info("Running greedy search ...")
        greedy = GreedySearch(
            layer_names=layer_names,
            sensitivity=sensitivity,
            cost_model=cost_model,
            proxy=proxy,
            skip_patterns=skip_patterns,
        )
        candidate_policies["greedy"] = greedy.search(
            bops_budget_ratio=args.bops_budget,
            max_sparsity=args.max_sparsity,
        )

    if args.search_strategy in ("both", "cmaes"):
        logger.info("Running CMA-ES search ...")
        cmaes = CMAESSearch(
            layer_names=layer_names,
            sensitivity=sensitivity,
            cost_model=cost_model,
            proxy=proxy,
            skip_patterns=skip_patterns,
        )
        candidate_policies["cmaes"] = cmaes.search(
            bops_budget_ratio=args.bops_budget,
            max_sparsity=args.max_sparsity,
            popsize=args.cmaes_popsize,
            max_generations=args.cmaes_generations,
            seed=args.seed,
        )

    best_policy, winning_strategy = select_best_policy(
        candidate_policies, layer_names, sensitivity, proxy, cost_model
    )
    logger.info("Winning strategy: %s", winning_strategy)

    # Save policy to disk
    policy_path = output_dir / "best_policy.json"
    with open(policy_path, "w") as fh:
        json.dump({"strategy": winning_strategy, "policy": best_policy}, fh, indent=2)
    logger.info("Best policy saved to %s", policy_path)

    # ------------------------------------------------------------------ #
    # 6. Apply pruning + KD recovery                                        #
    # ------------------------------------------------------------------ #
    # Extract per-layer Taylor scores from sensitivity dict
    taylor_scores = {
        name: data.get("taylor_scores", [])
        for name, data in sensitivity.items()
    }

    kd = KDRecovery(
        teacher=teacher,
        device=device,
        epochs=args.kd_epochs,
        temperature=4.0,
        alpha=0.9,
    )

    pruner = TaylorPruner(
        model=model,
        taylor_scores=taylor_scores,
        skip_patterns=skip_patterns,
    )

    # Build pruning-only policy (sparsity values only, bit-widths handled by AdaRound)
    pruning_policy = {
        name: entry["sparsity"]
        for name, entry in best_policy.items()
        if isinstance(entry, dict)
    }

    def recovery_fn(m):
        kd.recover(m, train_loader, val_loader)

    logger.info("Starting iterative pruning (%d rounds) ...", args.pruning_rounds)
    pruner.iterative_prune(
        policy=pruning_policy,
        n_rounds=args.pruning_rounds,
        recovery_fn=recovery_fn,
    )

    pruned_model = pruner.model
    ckpt_pruned = ckpt_dir / "checkpoint_pruned.pth"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": pruned_model.state_dict()}, ckpt_pruned)
    logger.info("Pruned model saved to %s", ckpt_pruned)

    # ------------------------------------------------------------------ #
    # 7. AdaRound on pruned model (Stage 2.5)                               #
    # ------------------------------------------------------------------ #
    if not args.skip_adaround:
        logger.info("Running AdaRound on pruned model ...")
        calib_inputs, _ = next(iter(train_loader))
        calib_inputs = calib_inputs.to(device)

        for name, module in pruned_model.named_modules():
            if name not in best_policy:
                continue
            if not isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                continue

            entry = best_policy[name]
            wb = entry.get("weight_bits", 8)
            if wb >= 32:
                continue

            opt = AdaRoundOptimizer(n_bits=wb, n_steps=args.adaround_steps)

            # For Conv2d we need to pass a compatible input slice
            with torch.no_grad():
                if isinstance(module, torch.nn.Conv2d):
                    layer_input = calib_inputs[: args.calib_batches]
                else:
                    # Linear: need flattened input — obtain via a partial forward
                    layer_input = calib_inputs[: args.calib_batches]

            opt.optimize_layer(module, layer_input, n_steps=args.adaround_steps)
            module.weight.data = opt.get_rounded_weights()

        logger.info("AdaRound complete.")
    else:
        logger.info("Skipping AdaRound (--skip-adaround set).")

    # ------------------------------------------------------------------ #
    # 8. MASE compiler pass (Stage 3)                                       #
    # ------------------------------------------------------------------ #
    logger.info("Running MASE compiler pass ...")
    mase_keys = qcfg.get("mase_keys", {})
    mase_gen = MaseConfigGenerator(
        weight_width_key=mase_keys.get("weight_width_key", "weight_width"),
        activation_width_key=mase_keys.get("activation_width_key", "data_in_width"),
        weight_frac_key=mase_keys.get("weight_frac_key", "weight_frac_width"),
        activation_frac_key=mase_keys.get("activation_frac_key", "data_in_frac_width"),
        default_frac_width=mase_keys.get("default_frac_width", 6),
    )
    # generate() takes bit_map as argument — strip sparsity key so it
    # matches BitMap schema: {layer: {"weight_bits": int, "activation_bits": int}}
    bit_map_for_mase = {
        name: {
            "weight_bits":     entry["weight_bits"],
            "activation_bits": entry["activation_bits"],
        }
        for name, entry in best_policy.items()
        if isinstance(entry, dict) and "weight_bits" in entry
    }
    quant_config = mase_gen.generate(bit_map_for_mase)
    quant_config_path = output_dir / "quant_config.json"
    mase_gen.save(quant_config, quant_config_path)
    logger.info("Quantization config saved to %s", quant_config_path)

    ckpt_quantized = ckpt_dir / "checkpoint_quantized.pth"
    torch.save({"model_state_dict": pruned_model.state_dict()}, ckpt_quantized)
    logger.info("Quantized checkpoint saved to %s", ckpt_quantized)

    # ------------------------------------------------------------------ #
    # 9. Resource parser + feedback (Stage 3.5)                             #
    # ------------------------------------------------------------------ #
    predicted_latency = cost_model.predict_latency_ms(best_policy)
    parser = ResourceParser(cost_model=cost_model, mase_output_dir=output_dir)
    summary = parser.parse_and_recalibrate(predicted_latency_ms=predicted_latency)
    if summary is not None:
        parser.save_summary(summary, output_dir / "resource_summary.json")

    logger.info("run_search.py complete.")
    logger.info("Outputs:")
    logger.info("  Best policy    : %s", policy_path)
    logger.info("  Pruned model   : %s", ckpt_pruned)
    logger.info("  Quant config   : %s", quant_config_path)
    logger.info("  Quant checkpoint: %s", ckpt_quantized)


if __name__ == "__main__":
    main()
