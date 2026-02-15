import csv
from copy import deepcopy
from pathlib import Path
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trl import run_training, run_backprop, run_local_contrastive
from trl.config.config import Config
from trl.config.configurations import (
    finish_setup,
    standard_setup,
    last_layer_head,
    enable_trace,
    enable_lateral_shift,
    enable_lateral_shift_cov_target,
)


def base_cfg(epochs: int, head_epochs: int):
    cfg = Config()
    cfg.logger = "csv"
    cfg.store_config.device = "cpu"
    cfg.data_config.num_workers = 0
    cfg.data_config.pin_memory = False
    cfg.epochs = epochs
    cfg.head_epochs = head_epochs
    return cfg


def run_variant(name: str, cfg: Config, seed: int = 42, runner: str = "trl"):
    cfg_local = deepcopy(cfg)
    cfg_local.seed = seed
    cfg_local.run_name = f"{cfg_local.run_name} {name}"
    finish_setup(cfg_local)
    if runner == "trl":
        val_acc = run_training.run(cfg_local)
    elif runner == "bp":
        val_acc = run_backprop.run(cfg_local)
    elif runner == "local_supcon":
        val_acc = run_local_contrastive.run(cfg_local)
    else:
        raise ValueError(f"Unknown runner: {runner}")
    return float(val_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--head_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace_decay_fast", type=float, default=0.5)
    args = parser.parse_args()

    results = []

    def trl_base():
        cfg = base_cfg(args.epochs, args.head_epochs)
        standard_setup(cfg)
        return cfg

    def trls_base():
        return base_cfg(args.epochs, args.head_epochs)

    variants = [
        # TRL family
        ("trl_baseline", lambda: trl_base(), "trl"),
        ("trl_last_layer_head", lambda: last_layer_head(trl_base()), "trl"),
        ("trl_trace", lambda: enable_trace(trl_base(), decay=0.9), "trl"),
        ("trl_lateral_shift", lambda: enable_lateral_shift(trl_base()), "trl"),
        ("trl_trace_fast", lambda: enable_trace(trl_base(), decay=args.trace_decay_fast), "trl"),
        ("trl_lateral_shift_cov_target", lambda: enable_lateral_shift_cov_target(enable_lateral_shift(trl_base())), "trl"),
        (
            "trl_trace_lateral_shift_fast_cov",
            lambda: enable_lateral_shift_cov_target(
                enable_lateral_shift(enable_trace(trl_base(), decay=args.trace_decay_fast))
            ),
            "trl",
        ),
        (
            "trl_trace_lateral_shift_last_layer",
            lambda: last_layer_head(
                enable_lateral_shift_cov_target(
                    enable_lateral_shift(enable_trace(trl_base(), decay=args.trace_decay_fast))
                )
            ),
            "trl",
        ),
        # TRL-S family
        ("trls_baseline", lambda: trls_base(), "trl"),
        ("trls_last_layer_head", lambda: last_layer_head(trls_base()), "trl"),
        (
            "trls_shift_shiftcov_last_layer",
            lambda: last_layer_head(
                enable_lateral_shift_cov_target(enable_lateral_shift(trls_base()))
            ),
            "trl",
        ),
        (
            "trls_tracefast_shift_shiftcov_last_layer",
            lambda: last_layer_head(
                enable_lateral_shift_cov_target(
                    enable_lateral_shift(enable_trace(trls_base(), decay=args.trace_decay_fast))
                )
            ),
            "trl",
        ),
        # Backprop baselines
        ("bp_all_layers_head", lambda: trls_base(), "bp"),
        ("bp_last_layer_head", lambda: last_layer_head(trls_base()), "bp"),
        # Local supervised contrastive baselines
        ("local_supcon_all_layers_head", lambda: trls_base(), "local_supcon"),
        ("local_supcon_last_layer_head", lambda: last_layer_head(trls_base()), "local_supcon"),
    ]

    for name, build_cfg_fn, runner in variants:
        cfg_variant = build_cfg_fn()
        acc = run_variant(name, cfg_variant, seed=args.seed, runner=runner)
        results.append({"variant": name, "val_acc": acc, "val_error": 1.0 - acc})
        print(f"{name}: {acc:.4f}")

    out_path = Path("analysis_outputs") / "trl_ablation_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "val_acc", "val_error"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results to {out_path}")
