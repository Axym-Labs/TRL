import json

import pytest

from terel.resubmission.analysis_v2 import analyze_v2_results


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
