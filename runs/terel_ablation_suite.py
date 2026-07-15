import csv
from copy import deepcopy
from pathlib import Path
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from terel import run_training, run_backprop, run_local_contrastive
from terel.config.config import Config
from terel.config.configurations import (
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


def run_variant(name: str, cfg: Config, seed: int = 42, runner: str = "terel"):
    cfg_local = deepcopy(cfg)
    cfg_local.seed = seed
    cfg_local.run_name = f"{cfg_local.run_name} {name}"
    finish_setup(cfg_local)
    if runner == "terel":
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

    def terel_base():
        cfg = base_cfg(args.epochs, args.head_epochs)
        standard_setup(cfg)
        return cfg

    def terels_base():
        return base_cfg(args.epochs, args.head_epochs)

    variants = [
        # TeReL family
        ("terel_baseline", lambda: terel_base(), "terel"),
        ("terel_last_layer_head", lambda: last_layer_head(terel_base()), "terel"),
        ("terel_trace", lambda: enable_trace(terel_base(), decay=0.9), "terel"),
        ("terel_lateral_shift", lambda: enable_lateral_shift(terel_base()), "terel"),
        ("terel_trace_fast", lambda: enable_trace(terel_base(), decay=args.trace_decay_fast), "terel"),
        ("terel_lateral_shift_cov_target", lambda: enable_lateral_shift_cov_target(enable_lateral_shift(terel_base())), "terel"),
        (
            "terel_trace_lateral_shift_fast_cov",
            lambda: enable_lateral_shift_cov_target(
                enable_lateral_shift(enable_trace(terel_base(), decay=args.trace_decay_fast))
            ),
            "terel",
        ),
        (
            "terel_trace_lateral_shift_last_layer",
            lambda: last_layer_head(
                enable_lateral_shift_cov_target(
                    enable_lateral_shift(enable_trace(terel_base(), decay=args.trace_decay_fast))
                )
            ),
            "terel",
        ),
        # TeReL-S family
        ("terels_baseline", lambda: terels_base(), "terel"),
        ("terels_last_layer_head", lambda: last_layer_head(terels_base()), "terel"),
        (
            "terels_shift_shiftcov_last_layer",
            lambda: last_layer_head(
                enable_lateral_shift_cov_target(enable_lateral_shift(terels_base()))
            ),
            "terel",
        ),
        (
            "terels_tracefast_shift_shiftcov_last_layer",
            lambda: last_layer_head(
                enable_lateral_shift_cov_target(
                    enable_lateral_shift(enable_trace(terels_base(), decay=args.trace_decay_fast))
                )
            ),
            "terel",
        ),
        # Backprop baselines
        ("bp_all_layers_head", lambda: terels_base(), "bp"),
        ("bp_last_layer_head", lambda: last_layer_head(terels_base()), "bp"),
        # Local supervised contrastive baselines
        ("local_supcon_all_layers_head", lambda: terels_base(), "local_supcon"),
        ("local_supcon_last_layer_head", lambda: last_layer_head(terels_base()), "local_supcon"),
    ]

    for name, build_cfg_fn, runner in variants:
        cfg_variant = build_cfg_fn()
        acc = run_variant(name, cfg_variant, seed=args.seed, runner=runner)
        results.append({"variant": name, "val_acc": acc, "val_error": 1.0 - acc})
        print(f"{name}: {acc:.4f}")

    out_path = Path("analysis_outputs") / "terel_ablation_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "val_acc", "val_error"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results to {out_path}")
