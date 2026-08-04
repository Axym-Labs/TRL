"""Analyze the frozen normalization-matched random control."""

import argparse
import json
from pathlib import Path

from .statistics import paired_contrast, summarize_values


def _load_records(root, run_id, seeds):
    records = []
    for seed in seeds:
        path = Path(root) / "mnist" / run_id / f"seed-{seed}.json"
        if not path.exists():
            raise ValueError(f"missing control record: {path}")
        record = json.loads(path.read_text())
        identity = (
            record.get("dataset"),
            record.get("run_id"),
            int(record.get("seed", -1)),
        )
        if identity != ("mnist", run_id, int(seed)):
            raise ValueError(f"mismatched control record: {path}")
        records.append(record)
    return records


def _values_by_seed(records):
    return {
        int(record["seed"]): float(record["metrics"]["accuracy"])
        for record in records
    }


def _validate_calibrated_random(records):
    for record in records:
        if record.get("encoder_training") is not None:
            raise ValueError("calibrated random control must not train encoder parameters")
        if record.get("encoder_config", {}).get("normalization") != "batch_norm":
            raise ValueError("calibrated random control must use BatchNorm")
        calibration = record.get("normalization_calibration")
        if not calibration or int(calibration.get("passes", 0)) != 1:
            raise ValueError("calibrated random control is missing its frozen calibration")
        if int(calibration.get("examples", 0)) <= 0 or int(calibration.get("batches", 0)) <= 0:
            raise ValueError("calibration accounting is incomplete")


def analyze_normalization_control(
    control_root,
    reference_root,
    *,
    seeds=(1101, 1202, 1303, 1404, 1505),
    control_id="random-bn-calibrated-all",
    terel_id="terel-all",
    unnormalized_id="random-all",
):
    seeds = tuple(int(seed) for seed in seeds)
    calibrated_records = _load_records(control_root, control_id, seeds)
    terel_records = _load_records(reference_root, terel_id, seeds)
    unnormalized_records = _load_records(reference_root, unnormalized_id, seeds)
    _validate_calibrated_random(calibrated_records)

    calibrated = _values_by_seed(calibrated_records)
    terel = _values_by_seed(terel_records)
    unnormalized = _values_by_seed(unnormalized_records)
    terel_contrast = paired_contrast(terel, calibrated)
    return {
        "schema_version": 4,
        "seeds": list(seeds),
        "random_bn_calibrated": summarize_values(list(calibrated.values())),
        "random_no_normalization": summarize_values(list(unnormalized.values())),
        "terel": summarize_values(list(terel.values())),
        "terel_minus_random_bn": terel_contrast,
        "random_bn_minus_random_no_norm": paired_contrast(calibrated, unnormalized),
        "decision": "supported" if terel_contrast["mean_difference"] > 0 else "rejected",
        "calibration": calibrated_records[0]["normalization_calibration"],
    }


def _interval(summary):
    return f'{summary["mean"] * 100:.2f} $\\pm$ {summary["sample_sd"] * 100:.2f}'


def render_normalization_control_latex(analysis):
    primary = analysis["terel_minus_random_bn"]
    normalization = analysis["random_bn_minus_random_no_norm"]
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{MNIST accuracy of the normalization-matched random control across five paired seeds. Hidden weights and affine BatchNorm parameters remain at initialization; only running statistics are calibrated on the training split. Method rows report mean $\pm$ sample standard deviation, and contrasts report 95\% Student-$t$ intervals.}",
            r"\label{tab:normalization-control}",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"Method or contrast & Accuracy / difference \\",
            r"\midrule",
            f'Random, no normalization & {_interval(analysis["random_no_normalization"])} \\\\',
            f'BatchNorm-calibrated random & {_interval(analysis["random_bn_calibrated"])} \\\\',
            f'\\TeReL{{}} & {_interval(analysis["terel"])} \\\\',
            r"\midrule",
            f'\\TeReL{{}} $-$ calibrated random & {primary["mean_difference"] * 100:.2f} '
            f'[{primary["student_t_ci95_low"] * 100:.2f}, {primary["student_t_ci95_high"] * 100:.2f}] \\\\',
            f'calibrated $-$ no-normalization random & {normalization["mean_difference"] * 100:.2f} '
            f'[{normalization["student_t_ci95_low"] * 100:.2f}, {normalization["student_t_ci95_high"] * 100:.2f}] \\\\',
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-results", required=True)
    parser.add_argument("--reference-results", required=True)
    parser.add_argument("--analysis-output", required=True)
    parser.add_argument("--results-tex", required=True)
    arguments = parser.parse_args(argv)
    analysis = analyze_normalization_control(
        arguments.control_results,
        arguments.reference_results,
    )
    Path(arguments.analysis_output).write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n"
    )
    Path(arguments.results_tex).write_text(render_normalization_control_latex(analysis))


if __name__ == "__main__":
    main()
