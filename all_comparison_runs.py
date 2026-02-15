import json
import traceback
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch

from trl import run_backprop
from trl import run_local_contrastive
from trl import run_training
from trl.config.configurations import *


OUTPUT_DIR = Path("output") / "metrics"


def apply_temporal_coherence(conf: Config, enabled: bool) -> Config:
    conf.data_config.coherence = 1.0 if enabled else 0.0
    return conf


def base_cfg() -> Config:
    cfg = Config()
    cfg.store_config.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.logger = "csv"
    return cfg


def identity_cfg(conf: Config) -> Config:
    return conf


def write_rows_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_with_seeds(
    cfg: Config,
    suite_name: str,
    setup_name: str,
    method_name: str,
    temporal_coherence: bool,
    runner=run_training.run,
    seed_start_at: int = 42,
    num_runs: int = 5,
) -> list[dict]:
    finish_setup(cfg)
    rows = []
    seeds = list(range(seed_start_at, seed_start_at + num_runs))
    for seed in seeds:
        cfg_current = deepcopy(cfg)
        cfg_current.seed = seed
        try:
            metrics = runner(cfg_current, return_metrics=True)
            val_metrics = metrics.get("validation_metrics", {})
            row = {
                "suite_name": suite_name,
                "setup_name": setup_name,
                "method_name": method_name,
                "temporal_coherence": temporal_coherence,
                "seed": seed,
                "primary_metric_name": metrics.get("primary_metric_name"),
                "primary_metric": metrics.get("primary_metric"),
                "run_name": cfg_current.run_name,
            }
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            rows.append(row)
        except Exception:
            rows.append({
                "suite_name": suite_name,
                "setup_name": setup_name,
                "method_name": method_name,
                "temporal_coherence": temporal_coherence,
                "seed": seed,
                "run_name": cfg_current.run_name,
                "error": traceback.format_exc().strip(),
            })
    return rows


def run_full_suite(
    suite_name: str,
    method_name: str,
    runner,
    cfg_factory,
    temporal_coherence: bool = True,
) -> list[dict]:
    setup_builders = [
        ("512_256", lambda conf: conf),
        ("512_256_aug_bn", aug_and_rbn_setup),
        ("eqprop_scale", eqprop_scale_network),
        ("ff_scale", ff_scale_network),
    ]

    suite_rows = []
    for setup_name, setup_fn in setup_builders:
        cfg = long_training(base_cfg())
        cfg = cfg_factory(cfg)
        cfg = apply_temporal_coherence(cfg, enabled=temporal_coherence)
        cfg = setup_fn(cfg)
        rows = run_with_seeds(
            cfg=cfg,
            suite_name=suite_name,
            setup_name=setup_name,
            method_name=method_name,
            temporal_coherence=temporal_coherence,
            runner=runner,
        )
        suite_rows.extend(rows)

    write_rows_csv(OUTPUT_DIR / f"{suite_name}.csv", suite_rows)
    return suite_rows


if __name__ == "__main__":
    all_rows = []

    all_rows.extend(run_full_suite(
        suite_name="trl_setup",
        method_name="trl",
        runner=run_training.run,
        cfg_factory=standard_setup,
        temporal_coherence=True,
    ))
    all_rows.extend(run_full_suite(
        suite_name="trls_setup",
        method_name="trl",
        runner=run_training.run,
        cfg_factory=identity_cfg,
        temporal_coherence=True,
    ))

    all_rows.extend(run_full_suite(
        suite_name="bp_coherence_on",
        method_name="bp",
        runner=run_backprop.run,
        cfg_factory=identity_cfg,
        temporal_coherence=True,
    ))
    all_rows.extend(run_full_suite(
        suite_name="bp_coherence_off",
        method_name="bp",
        runner=run_backprop.run,
        cfg_factory=identity_cfg,
        temporal_coherence=False,
    ))

    all_rows.extend(run_full_suite(
        suite_name="local_supcon_coherence_on",
        method_name="local_supcon",
        runner=run_local_contrastive.run,
        cfg_factory=identity_cfg,
        temporal_coherence=True,
    ))
    all_rows.extend(run_full_suite(
        suite_name="local_supcon_coherence_off",
        method_name="local_supcon",
        runner=run_local_contrastive.run,
        cfg_factory=identity_cfg,
        temporal_coherence=False,
    ))

    write_rows_csv(OUTPUT_DIR / "all_suites_concat.csv", all_rows)
    (OUTPUT_DIR / "all_suites_concat.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
