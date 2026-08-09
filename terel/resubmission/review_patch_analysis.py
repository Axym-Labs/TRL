"""Analyze the frozen validation evidence requested by the post-revision review."""

import argparse
import hashlib
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


def build_review_patch_validation_ledger(
    results_root,
    analysis,
    *,
    local_candidates=LOCAL_SUPCON_CANDIDATES,
    validation_seeds=(101, 202, 303),
    lagged_id="lagged-proxy-audit",
    direct_id="direct-covariance-matched",
):
    """Build the hash-linked ledger used to freeze the selected test run."""
    seeds = tuple(int(seed) for seed in validation_seeds)
    identifiers = tuple(local_candidates) + (direct_id, lagged_id)
    records = {}
    for identifier in identifiers:
        loaded = _load_records(results_root, identifier, seeds)
        paths = [Path(results_root) / identifier / f"seed-{seed}.json" for seed in seeds]
        summary = _accuracy_summary(loaded)
        records[identifier] = {
            "values": summary["raw"],
            "mean": summary["mean"],
            "sample_sd": summary["sample_sd"],
            "sha256": [
                hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
            ],
        }
    return {
        "schema_version": 3,
        "selection_complete": True,
        "selection_split": "mnist_validation",
        "selection_seeds": list(seeds),
        "primary_metric": "accuracy",
        "selected_primary": analysis["selected_local_supcon"],
        "selection_rule": analysis["selection_rule"],
        "records": records,
    }


def _load_confirmatory_records(root, run_id, seeds):
    records = []
    for seed in seeds:
        path = Path(root) / "mnist" / run_id / f"seed-{seed}.json"
        if not path.exists():
            raise ValueError(f"missing confirmatory record: {path}")
        record = json.loads(path.read_text())
        if (record.get("dataset"), record.get("run_id"), int(record.get("seed", -1))) != (
            "mnist",
            run_id,
            int(seed),
        ):
            raise ValueError(f"mismatched confirmatory record: {path}")
        records.append(record)
    return records


def analyze_confirmatory_comparator(
    comparator_root,
    reference_root,
    *,
    seeds=(1101, 1202, 1303, 1404, 1505),
    comparator_id="local-supcon-all",
    reference_id="terel-all",
):
    seeds = tuple(int(seed) for seed in seeds)
    local_records = _load_confirmatory_records(comparator_root, comparator_id, seeds)
    terel_records = _load_confirmatory_records(reference_root, reference_id, seeds)
    local_by_seed = {
        int(record["seed"]): float(record["metrics"]["accuracy"])
        for record in local_records
    }
    terel_by_seed = {
        int(record["seed"]): float(record["metrics"]["accuracy"])
        for record in terel_records
    }
    first_resource = local_records[0].get("resource_accounting", {})
    hidden_dims = local_records[0].get("encoder_config", {}).get("hidden_dims", [])
    normalization_bytes = 2 * sum(int(width) for width in hidden_dims) * 4 + len(hidden_dims) * 8
    return {
        "schema_version": 3,
        "seeds": list(seeds),
        "local_supcon": {
            **summarize_values(list(local_by_seed.values())),
            "mean_encoder_seconds": float(
                np.mean(
                    [record["encoder_training"]["seconds"] for record in local_records]
                )
            ),
            "parameter_bytes": int(first_resource.get("parameter_bytes", 0)),
            "optimizer_state_bytes": int(
                first_resource.get("optimizer_state_bytes", 0)
            ),
            "normalization_buffer_bytes": normalization_bytes,
            "inference_encoder_bytes": int(first_resource.get("parameter_bytes", 0))
            + normalization_bytes,
        },
        "terel": summarize_values(list(terel_by_seed.values())),
        "terel_minus_local_supcon": paired_contrast(terel_by_seed, local_by_seed),
    }


def render_confirmatory_latex(analysis):
    local = analysis["local_supcon"]
    terel = analysis["terel"]
    contrast = analysis["terel_minus_local_supcon"]
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{MNIST test accuracy for the matched label-aware comparison across five seeds. Local SupCon uses all same-label positives in each minibatch. Both methods stop credit at layer boundaries and use the same architecture, 60 data presentations, and concatenated-layer probe.}",
            r"\label{tab:local-supcon-comparison}",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"Method & Accuracy \\",
            r"\midrule",
            f'\\TeReLBatched{{}} & {terel["mean"] * 100:.2f} $\\pm$ {terel["sample_sd"] * 100:.2f} \\\\',
            f'Local SupCon & {local["mean"] * 100:.2f} $\\pm$ {local["sample_sd"] * 100:.2f} \\\\',
            r"\midrule",
            f'\\TeReLBatched{{}} $-$ Local SupCon & {contrast["mean_difference"] * 100:.2f} '
            f'[{contrast["student_t_ci95_low"] * 100:.2f}, '
            f'{contrast["student_t_ci95_high"] * 100:.2f}] \\\\',
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip\footnotesize The bracketed value is a paired 95\% Student-$t$ interval.",
            r"\end{table}",
            "",
        ]
    )


def render_validation_appendix_latex(validation, confirmatory):
    candidate_rows = []
    for candidate in LOCAL_SUPCON_CANDIDATES:
        summary = validation["local_supcon_candidates"][candidate]
        config = summary["encoder_config"]
        label = (
            f'{float(config["learning_rate"]):.0e} & '
            f'{float(config["contrastive_temperature"]):.1f}'
        )
        value = f'{summary["mean"] * 100:.2f} $\\pm$ {summary["sample_sd"] * 100:.2f}'
        if candidate == validation["selected_local_supcon"]:
            value = r"\textbf{" + value + "}"
        candidate_rows.append(f"{label} & {value} " + r"\\")
    local = confirmatory["local_supcon"]
    raw = ", ".join(f"{value * 100:.2f}" for value in local["raw"])
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\footnotesize",
            r"\caption{Local SupCon selection and final record. The left block reports the validation grid; bold marks the configuration selected before final evaluation. The right block reports the resource state of that configuration.}",
            r"\label{tab:local-supcon-record}",
            r"\begin{minipage}[t]{0.48\linewidth}",
            r"\centering",
            r"\textbf{Validation selection}\par\smallskip",
            r"\begin{tabular}{rrc}",
            r"\toprule",
            r"Learning rate & Temperature & Accuracy \\",
            r"\midrule",
            *candidate_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{minipage}\hfill",
            r"\begin{minipage}[t]{0.48\linewidth}",
            r"\centering",
            r"\textbf{Final record}\par\smallskip",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"Quantity & Value \\",
            r"\midrule",
            f'Mean accuracy (\\%) & {local["mean"] * 100:.2f} $\\pm$ {local["sample_sd"] * 100:.2f} \\\\',
            f'Mean encoder seconds & {local["mean_encoder_seconds"]:.1f} \\\\',
            f'Model parameters (MiB) & {local["parameter_bytes"] / 2**20:.3f} \\\\',
            f'Optimizer state (MiB) & {local["optimizer_state_bytes"] / 2**20:.3f} \\\\',
            f'Normalization buffers (MiB) & {local["normalization_buffer_bytes"] / 2**20:.3f} \\\\',
            f'Inference encoder (MiB) & {local["inference_encoder_bytes"] / 2**20:.3f} \\\\',
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{minipage}",
            r"\par\smallskip",
            f"Final accuracies (\\%): {raw}.",
            r"\end{table}",
            "",
        ]
    )
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ledger-output")
    parser.add_argument("--confirmatory-results")
    parser.add_argument("--reference-results")
    parser.add_argument("--confirmatory-output")
    parser.add_argument("--results-tex")
    parser.add_argument("--appendix-tex")
    arguments = parser.parse_args(argv)
    analysis = analyze_validation_patch(arguments.validation_results)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    if arguments.ledger_output:
        ledger = build_review_patch_validation_ledger(
            arguments.validation_results, analysis
        )
        ledger_output = Path(arguments.ledger_output)
        ledger_output.parent.mkdir(parents=True, exist_ok=True)
        ledger_output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    confirmatory_arguments = (
        arguments.confirmatory_results,
        arguments.reference_results,
        arguments.confirmatory_output,
        arguments.results_tex,
        arguments.appendix_tex,
    )
    if any(confirmatory_arguments):
        if not all(confirmatory_arguments):
            parser.error("all confirmatory reporting arguments must be supplied together")
        confirmatory = analyze_confirmatory_comparator(
            arguments.confirmatory_results, arguments.reference_results
        )
        Path(arguments.confirmatory_output).write_text(
            json.dumps(confirmatory, indent=2, sort_keys=True) + "\n"
        )
        Path(arguments.results_tex).write_text(render_confirmatory_latex(confirmatory))
        Path(arguments.appendix_tex).write_text(
            render_validation_appendix_latex(analysis, confirmatory)
        )


if __name__ == "__main__":
    main()
