"""Analyze a final samplewise TeReL evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def _summary(values) -> dict:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or len(array) == 0:
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
    layer_count = len(records[0]["encoder_training"][field])
    if any(len(record["encoder_training"][field]) != layer_count for record in records):
        raise ValueError(f"inconsistent layer count for {field}")
    return [
        _summary(record["encoder_training"][field][layer] for record in records)
        for layer in range(layer_count)
    ]


def analyze_residual_state_evaluation(records: list[dict], validation_ledger: dict) -> dict:
    if len(records) != 5:
        raise ValueError("frozen residual-state evaluation requires five records")
    seeds = [int(record["seed"]) for record in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError("frozen residual-state evaluation requires distinct seeds")
    if any(record.get("evaluation_split") != "test" for record in records):
        raise ValueError("all final records must use the test evaluation split")
    if any(record.get("method") != "terel_residual" for record in records):
        raise ValueError("all final records must use samplewise TeReL")
    if not validation_ledger.get("selection_complete"):
        raise ValueError("validation selection must be complete")

    paired = validation_ledger["paired_accuracy"]
    ranks = validation_ledger["effective_rank"]
    return {
        "schema_version": 1,
        "final": {
            "seeds": seeds,
            "accuracy": _summary(record["metrics"]["accuracy"] for record in records),
            "effective_rank": _summary(
                record["representation_diagnostics"]["effective_rank"]
                for record in records
            ),
            "median_feature_variance": _summary(
                record["representation_diagnostics"]["median_feature_variance"]
                for record in records
            ),
            "base_state_rms": _layer_summaries(
                records, "base_residual_state_rms_mean"
            ),
            "settled_state_rms": _layer_summaries(
                records, "residual_state_rms_mean"
            ),
            "encoder_seconds": _summary(
                record["encoder_training"]["seconds"] for record in records
            ),
            "optimizer_steps": int(records[0]["encoder_training"]["optimizer_steps"]),
            "dynamic_state_numel": int(
                records[0]["encoder_training"]["dynamic_state_numel"]
            ),
            "causal_dynamic_state_numel": int(
                records[0]["encoder_training"].get(
                    "causal_dynamic_state_numel",
                    records[0]["encoder_training"]["dynamic_state_numel"],
                )
            ),
            "auxiliary_parameter_numel": int(
                records[0]["encoder_training"].get("auxiliary_parameter_numel", 0)
            ),
            "feedforward_parameter_numel": int(
                records[0]["encoder_training"].get(
                    "parameter_numel",
                    records[0]["resource_accounting"]["parameter_bytes"] // 4,
                )
            ),
            "resource_accounting": dict(records[0]["resource_accounting"]),
        },
        "validation": {
            "reference_accuracy": _summary(paired["terel_s_reference"]["values"]),
            "residual_accuracy": _summary(paired["terel_s_residual"]["values"]),
            "paired_accuracy_gain": _summary(
                paired["residual_minus_reference"]["values"]
            ),
            "reference_effective_rank": _summary(
                ranks["terel_s_reference"]["values"]
            ),
            "residual_effective_rank": _summary(
                ranks["terel_s_residual"]["values"]
            ),
        },
    }


def _percent(summary, digits=2) -> str:
    return f"{100 * summary['mean']:.{digits}f} $\\pm$ {100 * summary['sample_sd']:.{digits}f}"


def render_results_latex(analysis: dict) -> str:
    final = analysis["final"]
    validation = analysis["validation"]
    gain = validation["paired_accuracy_gain"]
    low, high = gain["student_t_ci95"]
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{Samplewise \TeReL{} evidence. Final accuracy is reported as mean $\pm$ sample standard deviation. The inhibition effect is paired on the validation split; its interval is a 95\% Student-$t$ interval.}",
            r"\label{tab:residual-state-primary}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Evidence role & Accuracy / effect (points) & Effective rank \\",
            r"\midrule",
            (
                r"Frozen final evaluation & "
                + _percent(final["accuracy"])
                + f" & {final['effective_rank']['mean']:.2f} $\\pm$ "
                + f"{final['effective_rank']['sample_sd']:.2f} \\\\"
            ),
            (
                r"Matched validation effect & "
                + f"{100 * gain['mean']:.2f} "
                + f"[{100 * low:.2f}, {100 * high:.2f}]"
                + f" & {validation['reference_effective_rank']['mean']:.2f} "
                + r"$\rightarrow$ "
                + f"{validation['residual_effective_rank']['mean']:.2f} \\\\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def render_appendix_latex(analysis: dict) -> str:
    final = analysis["final"]
    accuracy = ", ".join(f"{100 * value:.2f}" for value in final["accuracy"]["raw"])
    rank = ", ".join(f"{value:.2f}" for value in final["effective_rank"]["raw"])
    seeds = ", ".join(str(seed) for seed in final["seeds"])
    resources = final["resource_accounting"]
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{Raw final-run values for samplewise \TeReL{}. Accuracy is in percent.}",
            r"\label{tab:residual-state-raw}",
            r"\begin{tabular}{ll}",
            r"\toprule",
            r"Field & Values \\",
            r"\midrule",
            f"Seeds & {seeds} " + r"\\",
            f"Accuracy & {accuracy} " + r"\\",
            f"Effective rank & {rank} " + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{Training resources for samplewise \TeReL{}. Causal state, auxiliary parameters, feedforward parameters, and optimizer state are reported separately.}",
            r"\label{tab:residual-state-resources}",
            r"\begin{tabular}{rrrrr}",
            r"\toprule",
            r"Updates & Causal values & Auxiliary parameters & Feedforward parameters & Optimizer bytes \\",
            r"\midrule",
            (
                f"{final['optimizer_steps']:,} & "
                f"{final['causal_dynamic_state_numel']:,} & "
                f"{final['auxiliary_parameter_numel']:,} & "
                f"{final['feedforward_parameter_numel']:,} & "
                f"{resources['optimizer_state_bytes']:,} \\\\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def _write_text_atomic(path, text) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--validation-ledger", required=True)
    parser.add_argument("--analysis-output", required=True)
    parser.add_argument("--results-tex", required=True)
    parser.add_argument("--appendix-tex", required=True)
    arguments = parser.parse_args(argv)
    paths = sorted(
        (Path(arguments.results) / "mnist" / "terel-s-residual").glob("seed-*.json")
    )
    records = [json.loads(path.read_text()) for path in paths]
    validation = json.loads(Path(arguments.validation_ledger).read_text())
    analysis = analyze_residual_state_evaluation(records, validation)
    _write_text_atomic(
        arguments.analysis_output,
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(arguments.results_tex, render_results_latex(analysis))
    _write_text_atomic(arguments.appendix_tex, render_appendix_latex(analysis))


if __name__ == "__main__":
    main()
