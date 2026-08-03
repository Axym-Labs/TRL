"""Analyze the frozen post-confirmation TeReL mechanism audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import _write_text_atomic
from .statistics import paired_contrast, summarize_values


SEEDS = (101, 202, 303)
AUDIT_ORDER = ("no-temporal", "no-variance", "no-covariance", "shuffled-order")
LABELS = {
    "full": r"Full \TeReL{}",
    "no-temporal": r"$\lambda_S=0$",
    "no-variance": r"$\lambda_V=0$",
    "no-covariance": r"$\lambda_C=0$",
    "shuffled-order": "Shuffled order",
}


def _load_candidate(root: Path, candidate: str, seeds=SEEDS):
    return {
        int(seed): json.loads((root / candidate / f"seed-{seed}.json").read_text())
        for seed in seeds
    }


def _summarize_records(records, *, bootstrap_samples, bootstrap_seed):
    def values(field, nested=None):
        if nested is None:
            return [float(record[field]) for record in records.values()]
        return [float(record[field][nested]) for record in records.values()]

    return {
        "accuracy": summarize_values(
            values("metrics", "accuracy"),
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "effective_rank": summarize_values(
            values("representation_diagnostics", "effective_rank"),
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "median_feature_variance": summarize_values(
            values("representation_diagnostics", "median_feature_variance"),
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "nearest_centroid_accuracy": summarize_values(
            values("class_structure_diagnostics", "nearest_centroid_accuracy"),
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
    }


def analyze_mechanism_audit(
    reference_root,
    audit_root,
    *,
    reference_candidate="canonical-recovered-bn",
    seeds=SEEDS,
    bootstrap_samples=10_000,
    bootstrap_seed=260803,
):
    reference = _load_candidate(Path(reference_root), reference_candidate, seeds)
    reference_accuracy = {
        seed: float(record["metrics"]["accuracy"]) for seed, record in reference.items()
    }
    methods = {
        "full": _summarize_records(
            reference,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
    }
    for candidate in AUDIT_ORDER:
        records = _load_candidate(Path(audit_root), candidate, seeds)
        summary = _summarize_records(
            records,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        candidate_accuracy = {
            seed: float(record["metrics"]["accuracy"]) for seed, record in records.items()
        }
        summary["accuracy_difference_from_full"] = paired_contrast(
            candidate_accuracy,
            reference_accuracy,
            bootstrap_samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        methods[candidate] = summary
    return {
        "schema_version": 1,
        "seeds": list(seeds),
        "reference_candidate": reference_candidate,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "methods": methods,
    }


def render_latex(analysis):
    rows = []
    for name in ("full", *AUDIT_ORDER):
        record = analysis["methods"][name]
        accuracy = record["accuracy"]
        if name == "full":
            difference = "--"
        else:
            effect = record["accuracy_difference_from_full"]
            difference = f'{effect["mean_difference"] * 100:.2f}'
        rows.append(
            f'{LABELS[name]} & {accuracy["mean"] * 100:.2f} $\\pm$ '
            f'{accuracy["sample_sd"] * 100:.2f} & {difference} & '
            f'{record["effective_rank"]["mean"]:.1f} & '
            f'{record["median_feature_variance"]["mean"]:.3g} \\\\'
        )
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{Validation-only mechanism audit under the recovered canonical protocol (three paired seeds). Each intervention changes one factor. $\Delta$ is the accuracy difference from full \TeReL{} in percentage points.}",
            r"\label{tab:mechanism-audit-v2}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Intervention & Accuracy (\%) & $\Delta$ & $r_{\mathrm{eff}}$ & Median var. \\",
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
    parser.add_argument("--reference-results", required=True)
    parser.add_argument("--audit-results", required=True)
    parser.add_argument("--analysis-output", required=True)
    parser.add_argument("--results-tex", required=True)
    arguments = parser.parse_args(argv)
    analysis = analyze_mechanism_audit(
        arguments.reference_results,
        arguments.audit_results,
    )
    _write_text_atomic(
        arguments.analysis_output,
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(arguments.results_tex, render_latex(analysis))


if __name__ == "__main__":
    main()
