"""Analyze the frozen validation evidence requested by the post-revision review."""

import argparse
import json
from pathlib import Path

import numpy as np

from .statistics import paired_contrast, summarize_values


LOCAL_SUPCON_CANDIDATES = (
    "local-supcon-lr1e4-t01",
    "local-supcon-lr1e4-t02",
    "local-supcon-lr3e4-t01",
    "local-supcon-lr3e4-t02",
    "local-supcon-lr1e3-t01",
    "local-supcon-lr1e3-t02",
)


def _load_records(root, candidate, seeds):
    records = []
    for seed in seeds:
        path = Path(root) / candidate / f"seed-{seed}.json"
        if not path.exists():
            raise ValueError(f"missing validation record: {path}")
        record = json.loads(path.read_text())
        if (record.get("candidate_id"), int(record.get("seed", -1))) != (
            candidate,
            int(seed),
        ):
            raise ValueError(f"mismatched validation record: {path}")
        records.append(record)
    return records


def _accuracy_summary(records):
    return summarize_values([float(record["metrics"]["accuracy"]) for record in records])


def _layer_means(records, field):
    values = [record["encoder_training"][field] for record in records]
    width = len(values[0])
    if any(len(item) != width for item in values):
        raise ValueError(f"inconsistent layer count for {field}")
    return np.asarray(values, dtype=np.float64).mean(axis=0).tolist()


def analyze_validation_patch(
    results_root,
    *,
    local_candidates=LOCAL_SUPCON_CANDIDATES,
    validation_seeds=(101, 202, 303),
    lagged_id="lagged-proxy-audit",
    direct_id="direct-covariance-matched",
):
    seeds = tuple(int(seed) for seed in validation_seeds)
    candidates = {}
    candidate_records = {}
    for candidate in local_candidates:
        records = _load_records(results_root, candidate, seeds)
        candidate_records[candidate] = records
        candidates[candidate] = {
            **_accuracy_summary(records),
            "encoder_config": records[0]["encoder_config"],
        }

    def selection_key(candidate):
        summary = candidates[candidate]
        config = summary["encoder_config"]
        return (
            -summary["mean"],
            summary["sample_sd"],
            float(config["learning_rate"]),
            -float(config["contrastive_temperature"]),
        )

    selected = min(local_candidates, key=selection_key)
    lagged_records = _load_records(results_root, lagged_id, seeds)
    direct_records = _load_records(results_root, direct_id, seeds)
    lagged_by_seed = {
        int(record["seed"]): float(record["metrics"]["accuracy"])
        for record in lagged_records
    }
    direct_by_seed = {
        int(record["seed"]): float(record["metrics"]["accuracy"])
        for record in direct_records
    }
    return {
        "schema_version": 3,
        "validation_seeds": list(seeds),
        "selection_rule": (
            "highest mean validation accuracy; then lower sample SD, lower learning "
            "rate, and higher temperature"
        ),
        "selected_local_supcon": selected,
        "local_supcon_candidates": candidates,
        "lateral_controls": {
            "lagged_proxy": {
                **_accuracy_summary(lagged_records),
                "cosine_alignment_by_layer": _layer_means(
                    lagged_records, "lateral_proxy_cosine_mean"
                ),
                "relative_error_by_layer": _layer_means(
                    lagged_records, "lateral_proxy_relative_error_mean"
                ),
                "norm_ratio_by_layer": _layer_means(
                    lagged_records, "lateral_proxy_norm_ratio_mean"
                ),
                "audited_batches_by_layer": np.asarray(
                    [
                        record["encoder_training"]["lateral_proxy_audited_batches"]
                        for record in lagged_records
                    ],
                    dtype=np.int64,
                ).sum(axis=0).tolist(),
            },
            "direct_covariance": _accuracy_summary(direct_records),
            "direct_minus_lagged": paired_contrast(direct_by_seed, lagged_by_seed),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-results", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    analysis = analyze_validation_patch(arguments.validation_results)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
