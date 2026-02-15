import argparse
from copy import deepcopy
from functools import partial
from pathlib import Path
import os
import sys

import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trl import run_training
from trl.config.config import Config
from trl.config.configurations import finish_setup, standard_setup


def base_cfg(epochs: int, head_epochs: int) -> Config:
    cfg = Config()
    cfg.logger = "csv"
    cfg.store_config.device = "cpu"
    cfg.data_config.num_workers = 0
    cfg.data_config.pin_memory = False
    cfg.data_config.dataset_name = "mnist"
    cfg.epochs = epochs
    cfg.head_epochs = head_epochs
    return cfg


def latest_metrics_row(run_name: str) -> dict:
    logs_root = Path("lightning_logs") / run_name
    if not logs_root.exists():
        return {}
    versions = sorted(
        [p for p in logs_root.iterdir() if p.is_dir() and p.name.startswith("version_")],
        key=lambda p: p.stat().st_mtime,
    )
    if not versions:
        return {}
    df = pd.read_csv(versions[-1] / "metrics.csv")
    out = {}
    wanted = [
        "classifier_val_acc",
        "classifier_train_acc",
        "e0_layer_0/sim_loss",
        "e0_layer_0/std_loss",
        "e0_layer_0/cov_loss",
        "e0_layer_0/lateral_loss",
        "e0_layer_0/vicreg_loss",
        "e0_layer_1/sim_loss",
        "e0_layer_1/std_loss",
        "e0_layer_1/cov_loss",
        "e0_layer_1/lateral_loss",
        "e0_layer_1/vicreg_loss",
    ]
    for col in wanted:
        if col in df.columns and df[col].notna().any():
            out[col] = float(df[col].dropna().iloc[-1])
    if "epoch" in df.columns and df["epoch"].notna().any():
        out["epoch"] = int(df["epoch"].dropna().iloc[-1])
    return out


def set_sgd(cfg: Config, lr: float, momentum: float = 0.0):
    cfg.lr = lr
    cfg.encoder_optim = partial(torch.optim.SGD, momentum=momentum)


def make_variants():
    return [
        {
            "name": "adam_baseline",
            "apply": lambda c: None,
        },
        {
            "name": "sgd_plain_lr1e4",
            "apply": lambda c: set_sgd(c, lr=1e-4, momentum=0.0),
        },
        {
            "name": "sgd_plain_lr1e4_clip1",
            "apply": lambda c: (
                set_sgd(c, lr=1e-4, momentum=0.0),
                setattr(c, "encoder_grad_clip_norm", 1.0),
            ),
        },
        {
            "name": "sgd_plain_lr1e4_clip1_lat01",
            "apply": lambda c: (
                set_sgd(c, lr=1e-4, momentum=0.0),
                setattr(c, "encoder_grad_clip_norm", 1.0),
                setattr(c, "encoder_lat_lr_factor", 0.1),
            ),
        },
        {
            "name": "sgd_mom09_lr5e4_clip1",
            "apply": lambda c: (
                set_sgd(c, lr=5e-4, momentum=0.9),
                setattr(c, "encoder_grad_clip_norm", 1.0),
            ),
        },
        {
            "name": "sgd_mom09_lr5e4_clip1_lat01",
            "apply": lambda c: (
                set_sgd(c, lr=5e-4, momentum=0.9),
                setattr(c, "encoder_grad_clip_norm", 1.0),
                setattr(c, "encoder_lat_lr_factor", 0.1),
            ),
        },
    ]


def run_matrix(epochs: int, head_epochs: int, seed: int):
    results = []
    for variant in make_variants():
        cfg = base_cfg(epochs=epochs, head_epochs=head_epochs)
        standard_setup(cfg)
        variant["apply"](cfg)
        cfg.seed = seed
        cfg.run_name = f"opt_ablation_{variant['name']}"
        finish_setup(cfg)

        val_acc = run_training.run(cfg)
        row = {
            "variant": variant["name"],
            "seed": seed,
            "val_acc": float(val_acc),
            "val_error": 1.0 - float(val_acc),
            "encoder_optim": str(cfg.encoder_optim),
            "lr": cfg.lr,
            "use_cov_directly": bool(cfg.trloss_config.use_cov_directly),
            "encoder_grad_clip_norm": float(cfg.encoder_grad_clip_norm),
            "encoder_lat_lr_factor": float(cfg.encoder_lat_lr_factor),
        }
        row.update(latest_metrics_row(cfg.run_name))
        results.append(row)
        print(f"{variant['name']}: val_acc={float(val_acc):.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--head_epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = run_matrix(args.epochs, args.head_epochs, args.seed)
    out_dir = Path("output") / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "optimizer_ablation_matrix.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
