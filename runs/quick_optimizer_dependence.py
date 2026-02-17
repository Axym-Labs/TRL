import argparse
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


def set_sgd(cfg: Config, lr: float, momentum: float):
    cfg.lr = lr
    cfg.encoder_optim = partial(torch.optim.SGD, momentum=momentum)


def variants():
    return [
        {
            "name": "adam_joint_learned_lateral",
            "apply": lambda c: None,
        },
        {
            "name": "sgdm_joint_learned_lateral_lr5e4_m09_clip1",
            "apply": lambda c: (
                set_sgd(c, lr=5e-4, momentum=0.9),
                setattr(c, "encoder_grad_clip_norm", 1.0),
            ),
        },
        {
            "name": "sgdm_joint_learned_lateral_lr5e4_m09_clip1_lat01_bs256",
            "apply": lambda c: (
                set_sgd(c, lr=5e-4, momentum=0.9),
                setattr(c, "encoder_grad_clip_norm", 1.0),
                setattr(c, "encoder_lat_lr_factor", 0.1),
                setattr(c.data_config, "batch_size", 256),
            ),
        },
        {
            "name": "sgdm_joint_learned_lateral_lr3e4_m09_clip1_lat003",
            "apply": lambda c: (
                set_sgd(c, lr=3e-4, momentum=0.9),
                setattr(c, "encoder_grad_clip_norm", 1.0),
                setattr(c, "encoder_lat_lr_factor", 0.03),
            ),
        },
        {
            "name": "sgdm_joint_learned_lateral_lr3e4_m09_clip1_lat005",
            "apply": lambda c: (
                set_sgd(c, lr=3e-4, momentum=0.9),
                setattr(c, "encoder_grad_clip_norm", 1.0),
                setattr(c, "encoder_lat_lr_factor", 0.05),
            ),
        },
        {
            "name": "sgdm_joint_learned_lateral_lr3e4_m09_clip1_lat02",
            "apply": lambda c: (
                set_sgd(c, lr=3e-4, momentum=0.9),
                setattr(c, "encoder_grad_clip_norm", 1.0),
                setattr(c, "encoder_lat_lr_factor", 0.2),
            ),
        },
        {
            "name": "sgd_joint_learned_lateral_lr3e4_m00_clip1_lat003",
            "apply": lambda c: (
                set_sgd(c, lr=3e-4, momentum=0.0),
                setattr(c, "encoder_grad_clip_norm", 1.0),
                setattr(c, "encoder_lat_lr_factor", 0.03),
            ),
        },
        {
            "name": "sgd_joint_learned_lateral_lr3e4_m00_clip1_lat005",
            "apply": lambda c: (
                set_sgd(c, lr=3e-4, momentum=0.0),
                setattr(c, "encoder_grad_clip_norm", 1.0),
                setattr(c, "encoder_lat_lr_factor", 0.05),
            ),
        },
        {
            "name": "sgd_joint_learned_lateral_lr3e4_m00_clip1_lat02",
            "apply": lambda c: (
                set_sgd(c, lr=3e-4, momentum=0.0),
                setattr(c, "encoder_grad_clip_norm", 1.0),
                setattr(c, "encoder_lat_lr_factor", 0.2),
            ),
        },
        {
            "name": "sgd_joint_learned_lateral_lr1e4_m00_clip1_lat003",
            "apply": lambda c: (
                set_sgd(c, lr=1e-4, momentum=0.0),
                setattr(c, "encoder_grad_clip_norm", 1.0),
                setattr(c, "encoder_lat_lr_factor", 0.03),
            ),
        },
        {
            "name": "sgd_joint_learned_lateral_lr3e4_m00_clip1_lat01",
            "apply": lambda c: (
                set_sgd(c, lr=3e-4, momentum=0.0),
                setattr(c, "encoder_grad_clip_norm", 1.0),
                setattr(c, "encoder_lat_lr_factor", 0.1),
            ),
        },
        {
            "name": "sgdm_joint_learned_lateral_lr3e4_m09_clip1_lat01",
            "apply": lambda c: (
                set_sgd(c, lr=3e-4, momentum=0.9),
                setattr(c, "encoder_grad_clip_norm", 1.0),
                setattr(c, "encoder_lat_lr_factor", 0.1),
            ),
        },
        {
            "name": "sgdm_joint_learned_lateral_lr5e4_m09_clip1_lat05",
            "apply": lambda c: (
                set_sgd(c, lr=5e-4, momentum=0.9),
                setattr(c, "encoder_grad_clip_norm", 1.0),
                setattr(c, "encoder_lat_lr_factor", 0.5),
            ),
        },
    ]


def latest_metrics(run_name: str) -> dict:
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
        "e0_layer_0/cov_loss",
        "e0_layer_0/lateral_loss",
        "e0_layer_0/std_loss",
        "e0_layer_0/sim_loss",
        "e0_layer_1/cov_loss",
        "e0_layer_1/lateral_loss",
        "e0_layer_1/std_loss",
        "e0_layer_1/sim_loss",
    ]
    for col in wanted:
        if col in df.columns and df[col].notna().any():
            out[col] = float(df[col].dropna().iloc[-1])
    return out


def run_quick(seed: int, epochs: int, head_epochs: int, out_name: str, only_variants: set[str] | None = None):
    rows = []
    for v in variants():
        if only_variants is not None and v["name"] not in only_variants:
            continue
        cfg = base_cfg(epochs=epochs, head_epochs=head_epochs)
        standard_setup(cfg)
        v["apply"](cfg)
        cfg.seed = seed
        cfg.run_name = f"{out_name}_{v['name']}_e{epochs}_s{seed}"
        finish_setup(cfg)

        val_acc = run_training.run(cfg)
        row = {
            "variant": v["name"],
            "seed": seed,
            "epochs": epochs,
            "head_epochs": head_epochs,
            "batch_size": int(cfg.data_config.batch_size),
            "val_acc": float(val_acc),
            "val_error": 1.0 - float(val_acc),
            "encoder_optim": str(cfg.encoder_optim),
            "lr": cfg.lr,
            "encoder_grad_clip_norm": float(cfg.encoder_grad_clip_norm),
            "encoder_lat_lr_factor": float(cfg.encoder_lat_lr_factor),
            "use_cov_directly": bool(cfg.trloss_config.use_cov_directly),
            "lat_coeff": float(cfg.trloss_config.lat_coeff),
        }
        row.update(latest_metrics(cfg.run_name))
        rows.append(row)
        print(f"{v['name']}: val_acc={float(val_acc):.4f}")

    out_dir = Path("output") / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_name}_e{epochs}_s{seed}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--head_epochs", type=int, default=3)
    parser.add_argument("--out_name", type=str, default="optimizer_dependence_quick")
    parser.add_argument("--variants", type=str, default="")
    args = parser.parse_args()
    only_variants = None
    if args.variants.strip():
        only_variants = {v.strip() for v in args.variants.split(",") if v.strip()}
    run_quick(seed=args.seed, epochs=args.epochs, head_epochs=args.head_epochs, out_name=args.out_name, only_variants=only_variants)


if __name__ == "__main__":
    main()
