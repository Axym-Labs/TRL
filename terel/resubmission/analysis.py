"""Analyze manifest-locked confirmatory records and generate paper tables."""

import argparse
import json
from pathlib import Path

from .statistics import paired_contrast, summarize_values


PRIMARY_METRICS = {"mnist": "accuracy", "pamap2": "macro_f1"}
PRIMARY_CONTRASTS = {
    "mnist_terel_minus_random": ("mnist", "terel-local", "random"),
    "pamap2_ordered_minus_shuffled": (
        "pamap2",
        "terel-ordered",
        "terel-shuffled",
    ),
}
METHOD_LABELS = {
    "terel-local": r"\TeReL{}-local",
    "terel-ordered": r"\TeReL{}-ordered",
    "terel-shuffled": r"\TeReL{}-shuffled",
    "random": "Random",
    "local-supcon": "Local SupCon",
    "bp": "Supervised BP",
    "direct-covariance": "Direct covariance",
    "batch-sfa": "Batch SFA",
    "incremental-sfa": "IncSFA",
}


def _write_text_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _load_records(results_directory):
    records = {}
    for path in sorted(Path(results_directory).glob("*/*/seed-*.json")):
        result = json.loads(path.read_text())
        identity = (result.get("dataset"), result.get("run_id"), int(result.get("seed")))
        if None in identity[:2] or identity in records:
            raise ValueError(f"invalid or duplicate result identity in {path}")
        records[identity] = result
    return records


def _summaries_by_field(seed_records, field, *, bootstrap_samples, bootstrap_seed):
    keys = set.intersection(*(set(record.get(field, {})) for record in seed_records.values()))
    summaries = {}
    for key in sorted(keys):
        values = [record[field][key] for record in seed_records.values()]
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            summaries[key] = summarize_values(
                values,
                bootstrap_samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
    return summaries


def analyze_confirmatory_results(
    results_directory,
    *,
    expected_seeds=(1001, 1002, 1003, 1004, 1005),
    bootstrap_samples=10_000,
    bootstrap_seed=260803,
):
    records = _load_records(results_directory)
    expected_seeds = tuple(int(seed) for seed in expected_seeds)
    datasets = {}
    for dataset_name, primary_metric in PRIMARY_METRICS.items():
        run_ids = sorted(
            {run_id for dataset, run_id, _ in records if dataset == dataset_name}
        )
        if not run_ids:
            continue
        methods = {}
        for run_id in run_ids:
            seed_records = {
                seed: records[(dataset_name, run_id, seed)]
                for seed in expected_seeds
                if (dataset_name, run_id, seed) in records
            }
            if tuple(seed_records) != expected_seeds:
                raise ValueError(
                    f"{dataset_name}/{run_id} does not contain exactly seeds {expected_seeds}"
                )
            values = {
                seed: float(record["metrics"][primary_metric])
                for seed, record in seed_records.items()
            }
            methods[run_id] = {
                **summarize_values(
                    list(values.values()),
                    bootstrap_samples=bootstrap_samples,
                    seed=bootstrap_seed,
                ),
                "by_seed": {str(seed): value for seed, value in values.items()},
                "classification_metrics": _summaries_by_field(
                    seed_records,
                    "metrics",
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                ),
                "representation_diagnostics": _summaries_by_field(
                    seed_records,
                    "representation_diagnostics",
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                ),
                "class_structure_diagnostics": _summaries_by_field(
                    seed_records,
                    "class_structure_diagnostics",
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                ),
                "resource_accounting_by_seed": {
                    str(seed): record.get("resource_accounting", {})
                    for seed, record in seed_records.items()
                },
                "encoder_training_by_seed": {
                    str(seed): record.get("encoder_training")
                    for seed, record in seed_records.items()
                },
            }
        datasets[dataset_name] = {"primary_metric": primary_metric, "methods": methods}

    contrasts = {}
    for name, (dataset_name, treatment, control) in PRIMARY_CONTRASTS.items():
        if dataset_name not in datasets:
            continue
        methods = datasets[dataset_name]["methods"]
        if treatment not in methods or control not in methods:
            continue
        contrasts[name] = paired_contrast(
            {int(seed): value for seed, value in methods[treatment]["by_seed"].items()},
            {int(seed): value for seed, value in methods[control]["by_seed"].items()},
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        contrasts[name].update(
            {"dataset": dataset_name, "treatment": treatment, "control": control}
        )
    return {
        "schema_version": 1,
        "expected_seeds": list(expected_seeds),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "datasets": datasets,
        "primary_contrasts": contrasts,
    }


def _format_interval(summary):
    return (
        f'{summary["mean"]:.3f} $\\pm$ {summary["sample_sd"]:.3f} '
        f'[{summary["ci95_low"]:.3f}, {summary["ci95_high"]:.3f}]'
    )


def render_results_latex(analysis):
    blocks = []
    for dataset_name in ("mnist", "pamap2"):
        if dataset_name not in analysis["datasets"]:
            continue
        dataset = analysis["datasets"][dataset_name]
        title = "MNIST test accuracy" if dataset_name == "mnist" else "PAMAP2 test macro-F1"
        label = f"tab:{dataset_name}-confirmatory"
        rows = [
            f'{METHOD_LABELS.get(method, method)} & {_format_interval(summary)} \\\\'
            for method, summary in dataset["methods"].items()
        ]
        blocks.append(
            "\n".join(
                [
                    r"\begin{table}[t]",
                    r"\centering",
                    f"\\caption{{{title} across five seeds. Values are mean $\\pm$ sample SD "
                    "and percentile 95\\% bootstrap interval.}",
                    f"\\label{{{label}}}",
                    r"\begin{tabular}{lc}",
                    r"\toprule",
                    r"Method & Mean $\pm$ SD [95\% CI] \\",
                    r"\midrule",
                    *rows,
                    r"\bottomrule",
                    r"\end{tabular}",
                    r"\end{table}",
                ]
            )
        )
    contrast_rows = []
    for summary in analysis["primary_contrasts"].values():
        label = (
            f'{METHOD_LABELS.get(summary["treatment"], summary["treatment"])} '
            f'$-$ {METHOD_LABELS.get(summary["control"], summary["control"])}'
        )
        contrast_rows.append(
            f'{label} & {summary["mean_difference"]:.3f} '
            f'[{summary["ci95_low"]:.3f}, {summary["ci95_high"]:.3f}] \\\\'
        )
    if contrast_rows:
        blocks.append(
            "\n".join(
                [
                    r"\begin{table}[t]",
                    r"\centering",
                    r"\caption{Predeclared paired primary effects across matched seeds.}",
                    r"\label{tab:primary-contrasts}",
                    r"\begin{tabular}{lc}",
                    r"\toprule",
                    r"Contrast & Mean difference [95\% CI] \\",
                    r"\midrule",
                    *contrast_rows,
                    r"\bottomrule",
                    r"\end{tabular}",
                    r"\end{table}",
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


def render_appendix_latex(analysis):
    rows = []
    for dataset_name, dataset in analysis["datasets"].items():
        for method, summary in dataset["methods"].items():
            raw = ", ".join(f"{value:.4f}" for value in summary["raw"])
            rows.append(
                f'{dataset_name.upper()} & {METHOD_LABELS.get(method, method)} & {raw} \\\\'
            )
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{Raw confirmatory metrics in seed order 1001--1005.}",
            r"\label{tab:raw-confirmatory}",
            r"\begin{tabular}{lll}",
            r"\toprule",
            r"Dataset & Method & Raw seed values \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--analysis-output", required=True)
    parser.add_argument("--results-tex", required=True)
    parser.add_argument("--appendix-tex", required=True)
    arguments = parser.parse_args(argv)
    analysis = analyze_confirmatory_results(arguments.results)
    _write_text_atomic(
        arguments.analysis_output,
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(arguments.results_tex, render_results_latex(analysis))
    _write_text_atomic(arguments.appendix_tex, render_appendix_latex(analysis))


if __name__ == "__main__":
    main()
