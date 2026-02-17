import argparse
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import os
import sys

import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trl import run_backprop, run_training
from trl.config.config import Config
from trl.config.configurations import finish_setup, pamap2_setup, temporal_coherence_ordering
from trl.datasets.mnist import build_dataloaders


def base_cfg(epochs: int, seed: int):
    cfg = Config()
    cfg.logger = "csv"
    cfg.store_config.device = "cpu"
    cfg.data_config.num_workers = 0
    cfg.data_config.pin_memory = False
    cfg.seed = seed
    cfg.epochs = epochs
    cfg.head_epochs = epochs
    pamap2_setup(cfg)
    # Keep TRL-S comparisons on the same head policy (last layer only).
    cfg.head_use_layers = False
    return cfg


def latest_metrics(log_name: str):
    root = Path("lightning_logs") / log_name
    if not root.exists():
        return {}
    versions = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("version_")],
        key=lambda p: p.stat().st_mtime,
    )
    if not versions:
        return {}
    df = pd.read_csv(versions[-1] / "metrics.csv")
    out = {}

    def col_last(name):
        if name in df.columns and df[name].notna().any():
            return float(df[name].dropna().iloc[-1])
        return None

    def col_max(name):
        if name in df.columns and df[name].notna().any():
            return float(df[name].dropna().max())
        return None

    out["train_acc_last"] = col_last("train_acc")
    if out["train_acc_last"] is None:
        out["train_acc_last"] = col_last("classifier_train_acc")
    out["train_acc_best"] = col_max("train_acc")
    if out["train_acc_best"] is None:
        out["train_acc_best"] = col_max("classifier_train_acc")

    out["val_acc_last"] = col_last("val_acc")
    if out["val_acc_last"] is None:
        out["val_acc_last"] = col_last("classifier_val_acc")
    out["val_acc_best"] = col_max("val_acc")
    if out["val_acc_best"] is None:
        out["val_acc_best"] = col_max("classifier_val_acc")

    out["train_loss_last"] = col_last("train_loss")
    if out["train_loss_last"] is None:
        out["train_loss_last"] = col_last("classifier_train_loss")
    out["train_loss_best"] = col_max("train_loss")
    if out["train_loss_best"] is None:
        out["train_loss_best"] = col_max("classifier_train_loss")

    if "epoch" in df.columns and df["epoch"].notna().any():
        out["last_epoch_logged"] = int(df["epoch"].dropna().iloc[-1])
    return out


def run_variant(name: str, runner: str, cfg: Config):
    cfg_local = deepcopy(cfg)
    cfg_local.run_name = name
    finish_setup(cfg_local)

    # Data stats for report.
    train_loader, _, val_loader = build_dataloaders(cfg_local.data_config, cfg_local.problem_type)
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    input_dim = int(train_loader.dataset[0][0].numel())

    t0 = perf_counter()
    if runner == "bp":
        val_acc = run_backprop.run(cfg_local)
        log_name = f"{cfg_local.run_name}_bp"
    elif runner == "trls":
        val_acc = run_training.run(cfg_local)
        log_name = cfg_local.run_name
    else:
        raise ValueError(f"Unknown runner '{runner}'")
    dur = perf_counter() - t0

    row = {
        "variant": name,
        "runner": runner,
        "temporal_coherence_ordering": bool(cfg_local.data_config.temporal_coherence_ordering),
        "epochs": int(cfg_local.epochs),
        "seed": int(cfg_local.seed),
        "batch_size": int(cfg_local.data_config.batch_size),
        "lr": float(cfg_local.lr),
        "head_lr": float(cfg_local.head_lr),
        "encoder_optim": str(cfg_local.encoder_optim),
        "head_optim": str(cfg_local.head_optim),
        "head_use_layers": str(cfg_local.head_use_layers),
        "pamap2_stride": int(cfg_local.data_config.pamap2_stride),
        "pamap2_subject_disjoint_split": bool(cfg_local.data_config.pamap2_subject_disjoint_split),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "input_dim": input_dim,
        "val_acc_run_return": float(val_acc),
        "duration_sec": float(dur),
    }
    row.update(latest_metrics(log_name))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_name", type=str, default="pamap2_temporal_order_compare")
    args = parser.parse_args()

    rows = []

    # 1) BP with shuffled temporal data.
    cfg_bp_shuf = base_cfg(args.epochs, args.seed)
    temporal_coherence_ordering(cfg_bp_shuf, enabled=False)
    rows.append(run_variant("pamap2_bp_shuffled", "bp", cfg_bp_shuf))

    # 2) BP with ordered temporal data.
    cfg_bp_ord = base_cfg(args.epochs, args.seed)
    temporal_coherence_ordering(cfg_bp_ord, enabled=True)
    rows.append(run_variant("pamap2_bp_ordered", "bp", cfg_bp_ord))

    # 3) TRL-S with ordered temporal data.
    cfg_trls_ord = base_cfg(args.epochs, args.seed)
    temporal_coherence_ordering(cfg_trls_ord, enabled=True)
    rows.append(run_variant("pamap2_trls_ordered", "trls", cfg_trls_ord))

    out_dir = Path("output") / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}_e{args.epochs}_s{args.seed}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    for r in rows:
        print(
            f"{r['variant']}: val={r.get('val_acc_last', r['val_acc_run_return'])} "
            f"train={r.get('train_acc_last')} time={r['duration_sec']:.1f}s"
        )


if __name__ == "__main__":
    main()

