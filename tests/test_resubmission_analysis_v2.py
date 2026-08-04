import json

import pytest

from terel.resubmission.analysis_v2 import (
    analyze_v2_results,
    render_appendix_latex,
    render_main_latex,
)


def test_v2_analysis_reports_frozen_methods_and_paired_gaps(tmp_path):
    methods = {
        "terel-all": [0.97, 0.98],
        "terel-last": [0.96, 0.97],
        "terel-s-all": [0.95, 0.96],
        "random-all": [0.90, 0.91],
        "bp-all": [0.99, 1.00],
    }
    for method, values in methods.items():
        directory = tmp_path / "mnist" / method
        directory.mkdir(parents=True)
        for seed, value in zip((11, 22), values, strict=True):
            (directory / f"seed-{seed}.json").write_text(
                json.dumps(
                    {
                        "dataset": "mnist",
                        "run_id": method,
                        "seed": seed,
                        "metrics": {"accuracy": value},
                        "representation_diagnostics": {
                            "effective_rank": 30.0,
                            "median_feature_variance": 0.2,
                            "active_feature_fraction": 1.0,
                        },
                        "class_structure_diagnostics": {},
                        "resource_accounting": {},
                        "encoder_training": None,
                    }
                )
            )

    analysis = analyze_v2_results(tmp_path, expected_seeds=(11, 22))

    assert analysis["methods"]["terel-all"]["mean"] == pytest.approx(0.975)
    assert analysis["contrasts"]["terel-minus-bp"]["mean_difference"] == pytest.approx(-0.02)
    assert analysis["contrasts"]["terel-minus-random"]["mean_difference"] == pytest.approx(0.07)
    assert analysis["gates"]["noncollapse"] is True
    rendered = render_main_latex(analysis)
    assert "Random, no normalization (all layers)" in rendered
    assert r"\TeReL{} $-$ random, no normalization" in rendered
    assert r"\label{tab:mnist-primary}" in rendered
    assert rendered.count(r"\begin{table}[H]") == 2
    assert "Corrected MNIST" not in rendered


def test_v2_analysis_separates_training_state_from_inference_encoder_state(tmp_path):
    methods = ("terel-all", "terel-last", "terel-s-all", "random-all", "bp-all")
    for method in methods:
        directory = tmp_path / "mnist" / method
        directory.mkdir(parents=True)
        for seed in (11, 22):
            parameter_bytes = 296 if method == "bp-all" else 176
            (directory / f"seed-{seed}.json").write_text(
                json.dumps(
                    {
                        "dataset": "mnist",
                        "run_id": method,
                        "seed": seed,
                        "metrics": {"accuracy": 0.9},
                        "representation_diagnostics": {
                            "effective_rank": 30.0,
                            "median_feature_variance": 0.2,
                            "active_feature_fraction": 1.0,
                        },
                        "class_structure_diagnostics": {},
                        "encoder_config": {
                            "hidden_dims": [4, 2],
                            "normalization": "batch_norm" if method in ("terel-all", "terel-last", "bp-all") else "none",
                        },
                        "resource_accounting": {
                            "parameter_bytes": parameter_bytes,
                            "optimizer_state_bytes": 352,
                            "dynamic_state_bytes": 64,
                            "operation_proxy": {},
                        },
                        "encoder_training": None
                        if method == "random-all"
                        else {
                            "examples": 10,
                            "steps": 2,
                            "optimizer_steps": 2,
                            "seconds": 1.0,
                            "peak_device_memory_bytes": 1024,
                            "layer_parameter_delta_l2": [1.0, 1.0],
                        },
                    }
                )
            )

    analysis = analyze_v2_results(tmp_path, expected_seeds=(11, 22))

    terel = analysis["methods"]["terel-all"]["resources"]
    random = analysis["methods"]["random-all"]["resources"]
    bp = analysis["methods"]["bp-all"]["resources"]
    assert terel["inference_encoder_bytes"] == 240  # parameters + BN buffers/counters
    assert random["inference_encoder_bytes"] == 176
    assert random["encoder_examples"] == 0
    assert bp["inference_encoder_bytes"] == 240  # excludes the 2-by-10 supervised head
    rendered = render_appendix_latex(analysis)
    assert "Training-state decomposition" in rendered
    assert "Inference encoder" in rendered
    assert r"\label{tab:encoder-resources}" in rendered
    assert "Raw corrected MNIST" not in rendered
