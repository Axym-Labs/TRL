"""Analyze the frozen strengthening-2 result matrix."""

from __future__ import annotations

import json
from pathlib import Path

from .statistics import paired_contrast, summarize_values

DEFAULT_SUCCESSFUL_METHODS = (
    "terel",
    "terel-offline",
    "random",
    "supervised-linear",
    "supervised-offline",
    "layer-local-supcon",
    "batch-sfa",
    "incremental-sfa",
    "no-temporal",
    "no-variance",
    "shuffled-order",
)


def _load_records(results_directory: Path) -> dict[tuple[str, int], dict]:
    records = {}
    for path in sorted((results_directory / "mnist").glob("*/seed-*.json")):
        record = json.loads(path.read_text())
        identity = (str(record["run_id"]), int(record["seed"]))
        if identity in records:
            raise ValueError(f"duplicate result identity: {identity}")
        records[identity] = record
    return records


def _summary_by_seed(values: dict[int, float]) -> dict:
    return {
        **summarize_values([values[seed] for seed in sorted(values)]),
        "by_seed": {str(seed): values[seed] for seed in sorted(values)},
    }


def _numeric_diagnostic_summaries(seed_records: dict[int, dict]) -> dict:
    shared = set.intersection(
        *(set(record.get("representation_diagnostics", {})) for record in seed_records.values())
    )
    summaries = {}
    for name in sorted(shared):
        by_seed = {
            seed: record["representation_diagnostics"][name]
            for seed, record in seed_records.items()
        }
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in by_seed.values()
        ):
            summaries[name] = _summary_by_seed(by_seed)
    return summaries


def analyze_final_results(
    results_directory,
    *,
    expected_seeds=(1101, 1202, 1303, 1404, 1505),
    required_successful_methods=DEFAULT_SUCCESSFUL_METHODS,
    failure_ledger=None,
) -> dict:
    """Return seed summaries and matched effects from the frozen MNIST matrix."""
    results_directory = Path(results_directory)
    expected_seeds = tuple(int(seed) for seed in expected_seeds)
    records = _load_records(results_directory)
    methods = {}
    for run_id in required_successful_methods:
        seed_records = {
            seed: records[(run_id, seed)]
            for seed in expected_seeds
            if (run_id, seed) in records
        }
        if tuple(seed_records) != expected_seeds:
            raise ValueError(
                f"mnist/{run_id} does not contain exactly seeds {expected_seeds}"
            )
        accuracy = {
            seed: float(record["metrics"]["accuracy"])
            for seed, record in seed_records.items()
        }
        methods[run_id] = {
            "accuracy": _summary_by_seed(accuracy),
            "representation_diagnostics": _numeric_diagnostic_summaries(seed_records),
            "resource_accounting_by_seed": {
                str(seed): record.get("resource_accounting", {})
                for seed, record in seed_records.items()
            },
        }

    full_accuracy = {
        int(seed): value
        for seed, value in methods["terel"]["accuracy"]["by_seed"].items()
    }
    contrasts = {}
    for control in required_successful_methods:
        if control == "terel":
            continue
        control_accuracy = {
            int(seed): value
            for seed, value in methods[control]["accuracy"]["by_seed"].items()
        }
        contrasts[f"terel-minus-{control}"] = paired_contrast(
            full_accuracy, control_accuracy
        )

    paired_records = {
        seed: records[("terel", seed)]["paired_inference"]
        for seed in expected_seeds
    }
    if not all(
        record.get("same_trained_checkpoint") and record.get("same_fitted_probe")
        for record in paired_records.values()
    ):
        raise ValueError("online continuation comparison is not paired")
    online_accuracy_difference = {
        seed: float(record["accuracy_difference"])
        for seed, record in paired_records.items()
    }
    offline_accuracy = {
        seed: float(record["offline"]["metrics"]["accuracy"])
        for seed, record in paired_records.items()
    }
    online_accuracy = {
        seed: float(record["online"]["metrics"]["accuracy"])
        for seed, record in paired_records.items()
    }
    online_rank_difference = {
        seed: float(record["online"]["representation_diagnostics"]["effective_rank"])
        - float(record["offline"]["representation_diagnostics"]["effective_rank"])
        for seed, record in paired_records.items()
    }
    online_audit_records = {
        seed: records[("terel", seed)]["online_inference"]
        for seed in expected_seeds
    }
    failures = {}
    if failure_ledger is not None:
        failures = json.loads(Path(failure_ledger).read_text())

    return {
        "schema_version": 1,
        "dataset": "mnist",
        "expected_seeds": list(expected_seeds),
        "methods": methods,
        "contrasts": contrasts,
        "online_continuation": {
            "offline_accuracy": _summary_by_seed(offline_accuracy),
            "online_accuracy": _summary_by_seed(online_accuracy),
            "accuracy_difference": _summary_by_seed(online_accuracy_difference),
            "effective_rank_difference": _summary_by_seed(online_rank_difference),
            "parameter_delta_l2": _summary_by_seed(
                {
                    seed: float(record["parameter_delta_l2"])
                    for seed, record in online_audit_records.items()
                }
            ),
            "lateral_delta_l2": _summary_by_seed(
                {
                    seed: float(record["lateral_delta_l2"])
                    for seed, record in online_audit_records.items()
                }
            ),
            "mean_loss": _summary_by_seed(
                {
                    seed: float(record["mean_loss"])
                    for seed, record in online_audit_records.items()
                }
            ),
        },
        "failures": failures,
    }


def analyze_capture24_order(
    results_directory,
    *,
    expected_seeds=(501, 502, 503),
    role="validation",
) -> dict:
    """Summarize the validation-only CAPTURE-24 order intervention."""
    results_directory = Path(results_directory)
    expected_seeds = tuple(int(seed) for seed in expected_seeds)
    condition_names = (
        "chronological",
        "within-participant-shuffled",
        "random",
    )
    conditions = {}
    for condition in condition_names:
        records = {}
        for path in sorted((results_directory / condition).glob("seed-*.json")):
            record = json.loads(path.read_text())
            seed = int(record["seed"])
            if record.get("condition") != condition or seed in records:
                raise ValueError(f"invalid CAPTURE-24 order record: {path}")
            records[seed] = record
        if tuple(sorted(records)) != expected_seeds:
            raise ValueError(
                f"CAPTURE-24/{condition} does not contain exactly seeds "
                f"{expected_seeds}"
            )
        macro_f1 = {
            seed: float(records[seed]["metrics"]["macro_f1"])
            for seed in expected_seeds
        }
        conditions[condition] = {
            "macro_f1": _summary_by_seed(macro_f1),
            "representation_diagnostics": _numeric_diagnostic_summaries(records),
        }

    chronological = {
        int(seed): value
        for seed, value in conditions["chronological"]["macro_f1"]["by_seed"].items()
    }
    shuffled = {
        int(seed): value
        for seed, value in conditions["within-participant-shuffled"]["macro_f1"][
            "by_seed"
        ].items()
    }
    random = {
        int(seed): value
        for seed, value in conditions["random"]["macro_f1"]["by_seed"].items()
    }
    return {
        "schema_version": 1,
        "dataset": f"capture24-{role}",
        "expected_seeds": list(expected_seeds),
        "conditions": conditions,
        "contrasts": {
            "chronological-minus-shuffled": paired_contrast(
                chronological, shuffled
            ),
            "chronological-minus-random": paired_contrast(chronological, random),
        },
    }
