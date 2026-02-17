import argparse
from pathlib import Path
import os
import sys
from functools import partial

import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trl.config.config import Config
from trl.config.configurations import finish_setup
from trl import run_backprop


def base_cfg(epochs: int, seed: int) -> Config:
    cfg = Config()
    cfg.logger = "csv"
    cfg.store_config.device = "cpu"
    cfg.data_config.num_workers = 0
    cfg.data_config.pin_memory = False
    cfg.data_config.dataset_name = "mnist"
    cfg.problem_type = "pass"
    cfg.head_task = "classification"
    cfg.epochs = epochs
    cfg.head_epochs = epochs
    cfg.seed = seed
    return cfg


def run_compare(epochs: int, seed: int, out_name: str):
    rows = []
    momentum = 0.9
    for opt in ["adam", "sgdm", "sgd"]:
        cfg = base_cfg(epochs=epochs, seed=seed)
        if opt == "adam":
            cfg.encoder_optim = torch.optim.Adam
        elif opt == "sgdm":
            cfg.encoder_optim = partial(torch.optim.SGD, momentum=momentum)
        elif opt == "sgd":
            cfg.encoder_optim = partial(torch.optim.SGD, momentum=0.0)
        else:
            raise ValueError(f"Unsupported optimizer variant: {opt}")
        cfg.run_name = f"{out_name}_{opt}_e{epochs}_s{seed}"
        finish_setup(cfg)

        val_acc = run_backprop.run(cfg)
        rows.append(
            {
                "variant": f"bp_{opt}",
                "seed": seed,
                "epochs": epochs,
                "val_acc": float(val_acc),
                "val_error": 1.0 - float(val_acc),
                "lr": cfg.lr,
                "bp_optimizer": opt,
                "encoder_optim": str(cfg.encoder_optim),
                "bp_momentum": momentum if opt == "sgdm" else 0.0,
                "batch_size": int(cfg.data_config.batch_size),
            }
        )
        print(f"bp_{opt}: val_acc={float(val_acc):.4f}")

    out_dir = Path("output") / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_name}_e{epochs}_s{seed}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_name", type=str, default="backprop_optimizer_compare")
    args = parser.parse_args()
    run_compare(epochs=args.epochs, seed=args.seed, out_name=args.out_name)


if __name__ == "__main__":
    main()
