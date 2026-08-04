"""Analyze the frozen corrected-performance matrix and render paper tables."""

import argparse
import json
from pathlib import Path

from .analysis import _load_records, _write_text_atomic
from .statistics import paired_contrast, summarize_values


METHOD_ORDER = ("terel-all", "terel-last", "terel-s-all", "random-all", "bp-all")
METHOD_LABELS = {
    "terel-all": r"\TeReL{} (all layers)",
    "terel-last": r"\TeReL{} (last layer)",
    "terel-s-all": r"\TeReL{}-S (all layers)",
    "random-all": "Random, no normalization (all layers)",
    "bp-all": "Backpropagation (all layers)",
}
CONTRASTS = {
    "terel-minus-bp": ("terel-all", "bp-all"),
    "terel-minus-random": ("terel-all", "random-all"),
    "terel-last-minus-all": ("terel-last", "terel-all"),
    "terel-s-minus-random": ("terel-s-all", "random-all"),
}


def _numeric_summaries(records, field, *, bootstrap_samples, bootstrap_seed):
    common = set.intersection(*(set(record.get(field, {})) for record in records))
    summaries = {}
    for name in sorted(common):
        values = [record[field][name] for record in records]
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            summaries[name] = summarize_values(
                values, bootstrap_samples=bootstrap_samples, seed=bootstrap_seed
            )
    return summaries


def _resource_summary(records):
    training = [record.get("encoder_training") for record in records]
    training = [record for record in training if record is not None]
    resources = [record.get("resource_accounting", {}) for record in records]
    def mean_field(name, default=0.0):
        if not training:
            return float(default)
        return sum(float(record.get(name, default)) for record in training) / len(training)

    operation_values = []
    for resource in resources:
        operation = resource.get("operation_proxy", {})
        components = (
            operation.get("linear_forward_backward_mac_proxy"),
            operation.get("same_layer_pairwise_mac_proxy"),
        )
        operation_values.append(sum(value or 0 for value in components))
    first = resources[0]
    first_record = records[0]
    encoder = first_record.get("encoder_config", {})
    hidden_dims = [int(width) for width in encoder.get("hidden_dims", [])]
    normalization_buffer_bytes = 0
    if encoder.get("normalization") == "batch_norm":
        # Running mean/variance are float32; num_batches_tracked is int64.
        normalization_buffer_bytes = 2 * sum(hidden_dims) * 4 + len(hidden_dims) * 8
    parameter_bytes = int(first.get("parameter_bytes", 0))
    supervised_head_bytes = 0
    if first_record.get("method") == "bp" or first_record.get("run_id") == "bp-all":
        confusion_matrix = first_record.get("metrics", {}).get("confusion_matrix", [])
        num_classes = len(confusion_matrix) or 10
        if hidden_dims:
            supervised_head_bytes = (hidden_dims[-1] * num_classes + num_classes) * 4
    return {
        "encoder_examples": mean_field("examples"),
        "encoder_steps": mean_field("steps"),
        "optimizer_steps": mean_field("optimizer_steps", mean_field("steps")),
        "encoder_seconds": mean_field("seconds"),
        "peak_device_memory_bytes": mean_field("peak_device_memory_bytes"),
        "parameter_bytes": parameter_bytes,
        "dynamic_state_bytes": int(first.get("dynamic_state_bytes", 0)),
        "optimizer_state_bytes": int(first.get("optimizer_state_bytes", 0)),
        "normalization_buffer_bytes": normalization_buffer_bytes,
        "algorithmic_dynamic_state_bytes": max(
            int(first.get("dynamic_state_bytes", 0)) - normalization_buffer_bytes,
            0,
        ),
        "inference_encoder_bytes": (
            parameter_bytes - supervised_head_bytes + normalization_buffer_bytes
        ),
        "operation_proxy": sum(operation_values) / len(operation_values),
    }


def analyze_v2_results(
    results_directory,
    *,
    expected_seeds=(1101, 1202, 1303, 1404, 1505),
    bootstrap_samples=10_000,
    bootstrap_seed=260803,
):
    records = _load_records(results_directory)
    expected_seeds = tuple(int(seed) for seed in expected_seeds)
    methods = {}
    raw_by_method = {}
    record_by_method = {}
    for method in METHOD_ORDER:
        method_records = []
        by_seed = {}
        for seed in expected_seeds:
            identity = ("mnist", method, seed)
            if identity not in records:
                raise ValueError(f"missing frozen result {identity}")
            record = records[identity]
            method_records.append(record)
            by_seed[seed] = float(record["metrics"]["accuracy"])
        raw_by_method[method] = by_seed
        record_by_method[method] = method_records
        methods[method] = {
            **summarize_values(
                list(by_seed.values()),
                bootstrap_samples=bootstrap_samples,
                seed=bootstrap_seed,
            ),
            "by_seed": {str(seed): value for seed, value in by_seed.items()},
            "representation_diagnostics": _numeric_summaries(
                method_records,
                "representation_diagnostics",
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "class_structure_diagnostics": _numeric_summaries(
                method_records,
                "class_structure_diagnostics",
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "resources": _resource_summary(method_records),
        }

    contrasts = {
        name: paired_contrast(
            raw_by_method[treatment],
            raw_by_method[control],
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        for name, (treatment, control) in CONTRASTS.items()
    }
    terel_records = record_by_method["terel-all"]
    noncollapse = all(
        record["representation_diagnostics"]["effective_rank"] >= 20.0
        and record["representation_diagnostics"]["median_feature_variance"] >= 0.05
        and record["representation_diagnostics"]["active_feature_fraction"] >= 0.25
        for record in terel_records
    )
    layer_updates = all(
        record.get("encoder_training") is not None
        and all(
            delta > 0.0
            for delta in record["encoder_training"]["layer_parameter_delta_l2"]
        )
        for record in terel_records
    )
    gates = {
        "near_backprop": contrasts["terel-minus-bp"]["mean_difference"] >= -0.015,
        "learned_representation_benefit": contrasts["terel-minus-random"][
            "mean_difference"
        ]
        > 0.0,
        "noncollapse": noncollapse,
        "all_layers_updated": layer_updates,
    }
    return {
        "schema_version": 2,
        "expected_seeds": list(expected_seeds),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "methods": methods,
        "contrasts": contrasts,
        "gates": gates,
    }


def analyze_streaming_validation(
    recovery_root,
    *,
    candidate="stream-runningnorm-2ep",
    seeds=(101, 202, 303),
    bootstrap_samples=10_000,
    bootstrap_seed=260803,
):
    records = [
        json.loads((Path(recovery_root) / candidate / f"seed-{seed}.json").read_text())
        for seed in seeds
    ]
    values = [float(record["metrics"]["accuracy"]) for record in records]
    return {
        "candidate": candidate,
        **summarize_values(
            values, bootstrap_samples=bootstrap_samples, seed=bootstrap_seed
        ),
        "representation_diagnostics": _numeric_summaries(
            records,
            "representation_diagnostics",
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
        "resources_serial_seed_101": _resource_summary([records[0]]),
    }


def _interval(summary, *, scale=100.0, digits=2):
    return (
        f'{summary["mean"] * scale:.{digits}f} $\\pm$ '
        f'{summary["sample_sd"] * scale:.{digits}f}'
    )


def render_main_latex(analysis):
    rows = [
        f'{METHOD_LABELS[method]} & {_interval(analysis["methods"][method])} \\\\'
        for method in METHOD_ORDER
    ]
    contrast_rows = []
    for name, label in (
        ("terel-minus-bp", r"\TeReL{} $-$ BP"),
        ("terel-minus-random", r"\TeReL{} $-$ random, no normalization"),
        ("terel-last-minus-all", "last $-$ all layers"),
    ):
        summary = analysis["contrasts"][name]
        contrast_rows.append(
            f'{label} & {summary["mean_difference"] * 100:.2f} '
            f'[{summary["student_t_ci95_low"] * 100:.2f}, '
            f'{summary["student_t_ci95_high"] * 100:.2f}] \\\\'
        )
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{MNIST test accuracy across five seeds. Every method uses the same concatenated-layer readout, except the explicitly marked final-layer \TeReL{} row. Values report mean $\pm$ sample standard deviation.}",
            r"\label{tab:mnist-primary}",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"Method & Accuracy \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Paired differences in MNIST test accuracy. Values are percentage points with 95\% Student-$t$ intervals over five seed-paired differences.}",
            r"\label{tab:mnist-contrasts}",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"Contrast & Difference [95\% interval] \\",
            r"\midrule",
            *contrast_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def render_appendix_latex(analysis, streaming=None):
    raw_rows = []
    resource_rows = []
    state_rows = []
    for method in METHOD_ORDER:
        summary = analysis["methods"][method]
        raw = ", ".join(f"{value * 100:.2f}" for value in summary["raw"])
        raw_rows.append(f"{METHOD_LABELS[method]} & {raw} " + r"\\")
        resource = summary.get("resources")
        if resource is None:
            continue
        resource_rows.append(
            f'{METHOD_LABELS[method]} & {resource["encoder_examples"] / 1e6:.2f} & '
            f'{resource["optimizer_steps"]:.0f} & {resource["encoder_seconds"]:.1f} & '
            f'{resource["peak_device_memory_bytes"] / 2**20:.1f} & '
            f'{resource["operation_proxy"] / 1e9:.1f} \\\\'
        )
        state_rows.append(
            f'{METHOD_LABELS[method]} & {resource["parameter_bytes"] / 2**20:.3f} & '
            f'{resource["optimizer_state_bytes"] / 2**20:.3f} & '
            f'{resource["algorithmic_dynamic_state_bytes"] / 2**20:.3f} & '
            f'{resource["normalization_buffer_bytes"] / 2**20:.3f} & '
            f'{resource["inference_encoder_bytes"] / 2**20:.3f} \\\\'
        )
    streaming_block = ""
    if streaming is not None:
        diagnostics = streaming["representation_diagnostics"]
        streaming_block = "\n".join(
            [
                r"\begin{table}[H]",
                r"\centering",
                r"\caption{Samplewise \TeReL{}-S on the validation split across three seeds. The serial run provides the time and memory measurements; timings from concurrent repetitions are excluded.}",
                r"\label{tab:samplewise-streaming}",
                r"\begin{tabular}{lccc}",
                r"\toprule",
                r"Accuracy (\%) & Effective rank & Median variance & Encoder seconds \\",
                r"\midrule",
                f'{_interval(streaming)} & {diagnostics["effective_rank"]["mean"]:.1f} & '
                f'{diagnostics["median_feature_variance"]["mean"]:.3f} & '
                f'{streaming["resources_serial_seed_101"]["encoder_seconds"]:.1f} \\\\',
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        )
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{MNIST test accuracy for seeds 1101, 1202, 1303, 1404, and 1505, in that order.}",
            r"\label{tab:mnist-raw}",
            r"\begin{tabular}{ll}",
            r"\toprule",
            r"Method & Raw values \\",
            r"\midrule",
            *raw_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{Resource use during encoder fitting, averaged across five seeds. Examples count sample presentations. MAC denotes the declared linear and same-layer operation proxy, not energy use.}",
            r"\label{tab:encoder-resources}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Method & Examples (M) & Opt. steps & Seconds & Peak MiB & MAC (G) \\",
            r"\midrule",
            *resource_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{Training state in MiB. Model parameters include every parameter fitted during encoder training, including the supervised head for backpropagation. Algorithm state contains the \TeReL{} lateral and statistical state but excludes normalization buffers. Optimizer and algorithm state are training-only. The inference encoder retains hidden-layer parameters and normalization buffers, but discards the supervised head and \TeReL{} state.}",
            r"\label{tab:training-state-decomposition}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Method & Model params & Optimizer & Algorithm & Norm. & Inference encoder \\",
            r"\midrule",
            *state_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            streaming_block,
        ]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--streaming-results", required=True)
    parser.add_argument("--analysis-output", required=True)
    parser.add_argument("--results-tex", required=True)
    parser.add_argument("--appendix-tex", required=True)
    arguments = parser.parse_args(argv)
    analysis = analyze_v2_results(arguments.results)
    streaming = analyze_streaming_validation(arguments.streaming_results)
    analysis["streaming_validation"] = streaming
    _write_text_atomic(
        arguments.analysis_output,
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(arguments.results_tex, render_main_latex(analysis))
    _write_text_atomic(arguments.appendix_tex, render_appendix_latex(analysis, streaming))


if __name__ == "__main__":
    main()
