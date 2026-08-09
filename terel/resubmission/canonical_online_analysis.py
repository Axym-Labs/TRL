"""Analyze canonical one-pass TeReL final and matched validation records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def _summary(values) -> dict:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or not len(array):
        raise ValueError("summary values must be a nonempty vector")
    mean = float(array.mean())
    sample_sd = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    half_width = (
        float(stats.t.ppf(0.975, len(array) - 1) * sample_sd / np.sqrt(len(array)))
        if len(array) > 1
        else 0.0
    )
    return {
        "raw": [float(value) for value in array],
        "mean": mean,
        "sample_sd": sample_sd,
        "student_t_ci95": [mean - half_width, mean + half_width],
    }


def _layer_summaries(records, field) -> list[dict]:
    count = len(records[0]["encoder_training"][field])
    if any(len(record["encoder_training"][field]) != count for record in records):
        raise ValueError(f"inconsistent layer count for {field}")
    return [
        _summary(record["encoder_training"][field][layer] for record in records)
        for layer in range(count)
    ]


def _validate_records(records, *, expected_count, role):
    if len(records) != expected_count:
        raise ValueError(f"{role} requires {expected_count} records")
    seeds = [int(record["seed"]) for record in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{role} requires distinct seeds")
    if any(record.get("method") != "terel_residual" for record in records):
        raise ValueError(f"{role} records must use samplewise TeReL")
    return seeds


def analyze_canonical_online(final_records, inhibited_records, reference_records) -> dict:
    """Combine final evidence with the matched validation mechanism contrast."""
    final_seeds = _validate_records(final_records, expected_count=5, role="final evaluation")
    inhibited_seeds = _validate_records(
        inhibited_records, expected_count=3, role="inhibited validation"
    )
    reference_seeds = _validate_records(
        reference_records, expected_count=3, role="reference validation"
    )
    if inhibited_seeds != reference_seeds:
        raise ValueError("validation comparison requires matched seeds")

    training = final_records[0]["encoder_training"]
    resource = final_records[0]["resource_accounting"]
    hidden_dims = tuple(final_records[0]["encoder_config"]["hidden_dims"])
    if any(
        tuple(record["encoder_config"]["hidden_dims"]) != hidden_dims
        for record in final_records
    ):
        raise ValueError("final records must use the same hidden dimensions")
    dense_cells = int(training["examples"]) * sum(width * width for width in hidden_dims)
    inhibited_accuracy = np.asarray(
        [record["metrics"]["accuracy"] for record in inhibited_records]
    )
    reference_accuracy = np.asarray(
        [record["metrics"]["accuracy"] for record in reference_records]
    )
    return {
        "schema_version": 1,
        "final": {
            "seeds": final_seeds,
            "accuracy": _summary(record["metrics"]["accuracy"] for record in final_records),
            "macro_f1": _summary(record["metrics"]["macro_f1"] for record in final_records),
            "effective_rank": _summary(
                record["representation_diagnostics"]["effective_rank"]
                for record in final_records
            ),
            "base_neuron_state_rms": _layer_summaries(
                final_records, "base_residual_state_rms_mean"
            ),
            "inhibited_neuron_state_rms": _layer_summaries(
                final_records, "residual_state_rms_mean"
            ),
            "optimizer_steps": int(training["optimizer_steps"]),
            "causal_dynamic_state_numel": int(training["causal_dynamic_state_numel"]),
            "auxiliary_parameter_numel": int(training["auxiliary_parameter_numel"]),
            "feedforward_parameter_numel": int(training["parameter_numel"]),
            "optimizer_state_bytes": int(resource["optimizer_state_bytes"]),
            "same_layer_matrix_vector_mac_proxy": 2 * dense_cells,
            "same_layer_outer_product_mac_proxy": 2 * dense_cells,
        },
        "validation": {
            "seeds": inhibited_seeds,
            "inhibited_accuracy": _summary(inhibited_accuracy),
            "reference_accuracy": _summary(reference_accuracy),
            "paired_accuracy_gain": _summary(inhibited_accuracy - reference_accuracy),
            "inhibited_effective_rank": _summary(
                record["representation_diagnostics"]["effective_rank"]
                for record in inhibited_records
            ),
            "reference_effective_rank": _summary(
                record["representation_diagnostics"]["effective_rank"]
                for record in reference_records
            ),
        },
    }


def _records(path: Path) -> list[dict]:
    return [json.loads(item.read_text()) for item in sorted(path.glob("seed-*.json"))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--inhibited-validation", type=Path, required=True)
    parser.add_argument("--reference-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    analysis = analyze_canonical_online(
        _records(arguments.final),
        _records(arguments.inhibited_validation),
        _records(arguments.reference_validation),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    temporary.replace(arguments.output)


if __name__ == "__main__":
    main()
