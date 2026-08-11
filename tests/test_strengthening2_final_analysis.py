import json

import pytest

from terel.resubmission.strengthening2_analysis import (
    analyze_capture24_order,
    analyze_final_results,
    analyze_online_continuation,
)


def _write_result(root, run_id, seed, accuracy, *, online_accuracy=None):
    directory = root / "mnist" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": "mnist",
        "run_id": run_id,
        "seed": seed,
        "metrics": {"accuracy": accuracy},
        "representation_diagnostics": {
            "effective_rank": 80.0 + seed,
            "median_feature_variance": 0.5,
        },
        "resource_accounting": {"causal_dynamic_state_scalars": 12},
    }
    if online_accuracy is not None:
        record["online_inference"] = {
            "parameter_delta_l2": 3.0,
            "lateral_delta_l2": 20.0,
            "mean_loss": 0.8,
        }
        record["paired_inference"] = {
            "same_trained_checkpoint": True,
            "same_fitted_probe": True,
            "offline": {
                "metrics": {"accuracy": accuracy},
                "representation_diagnostics": {"effective_rank": 80.0 + seed},
            },
            "online": {
                "metrics": {"accuracy": online_accuracy},
                "representation_diagnostics": {"effective_rank": 20.0 + seed},
            },
            "accuracy_difference": online_accuracy - accuracy,
        }
    (directory / f"seed-{seed}.json").write_text(json.dumps(record))


def test_final_analysis_reports_paired_mechanisms_online_effect_and_failures(tmp_path):
    seeds = (11, 22)
    for seed, full, random, shuffled in zip(
        seeds, (0.96, 0.97), (0.90, 0.91), (0.93, 0.94), strict=True
    ):
        _write_result(
            tmp_path,
            "terel",
            seed,
            full,
            online_accuracy=full - 0.30,
        )
        _write_result(tmp_path, "random", seed, random)
        _write_result(tmp_path, "shuffled-order", seed, shuffled)
    failure_path = tmp_path / "failures.json"
    failure_path.write_text(
        json.dumps(
            {
                "no-covariance": {
                    "status": "non-finite",
                    "failed_seeds": [11, 22],
                }
            }
        )
    )

    analysis = analyze_final_results(
        tmp_path,
        expected_seeds=seeds,
        required_successful_methods=("terel", "random", "shuffled-order"),
        failure_ledger=failure_path,
    )

    assert analysis["methods"]["terel"]["accuracy"]["mean"] == pytest.approx(0.965)
    assert analysis["contrasts"]["terel-minus-random"][
        "mean_difference"
    ] == pytest.approx(0.06)
    assert analysis["contrasts"]["terel-minus-shuffled-order"][
        "mean_difference"
    ] == pytest.approx(0.03)
    assert analysis["online_continuation"]["accuracy_difference"][
        "mean"
    ] == pytest.approx(-0.30)
    assert analysis["online_continuation"]["online_accuracy"]["mean"] == pytest.approx(
        0.665
    )
    assert analysis["online_continuation"]["effective_rank_difference"][
        "mean"
    ] == pytest.approx(-60.0)
    assert analysis["online_continuation"]["parameter_delta_l2"][
        "mean"
    ] == pytest.approx(3.0)
    assert analysis["failures"]["no-covariance"]["status"] == "non-finite"


def test_final_analysis_rejects_incomplete_required_method(tmp_path):
    _write_result(tmp_path, "terel", 11, 0.96, online_accuracy=0.70)

    with pytest.raises(ValueError, match="does not contain exactly"):
        analyze_final_results(
            tmp_path,
            expected_seeds=(11, 22),
            required_successful_methods=("terel",),
        )


def test_capture24_order_analysis_uses_seed_paired_macro_f1(tmp_path):
    values = {
        "chronological": (0.55, 0.57),
        "within-participant-shuffled": (0.50, 0.51),
        "random": (0.33, 0.34),
    }
    for condition, scores in values.items():
        directory = tmp_path / condition
        directory.mkdir(parents=True)
        for seed, score in zip((11, 22), scores, strict=True):
            (directory / f"seed-{seed}.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "condition": condition,
                        "metrics": {"macro_f1": score},
                        "representation_diagnostics": {"effective_rank": 7.0},
                    }
                )
            )

    analysis = analyze_capture24_order(tmp_path, expected_seeds=(11, 22))

    effect = analysis["contrasts"]["chronological-minus-shuffled"]
    assert effect["mean_difference"] == pytest.approx(0.055)
    assert analysis["conditions"]["random"]["macro_f1"]["mean"] == pytest.approx(0.335)


def test_online_continuation_analysis_uses_separate_paired_results(tmp_path):
    for seed, offline, online in ((11, 0.96, 0.961), (22, 0.97, 0.969)):
        record = {
            "seed": seed,
            "online_inference": {
                "parameter_delta_l2": 0.05,
                "lateral_delta_l2": 22.0,
                "mean_loss": 0.8,
            },
            "paired_inference": {
                "same_trained_checkpoint": True,
                "same_fitted_probe": True,
                "offline": {
                    "metrics": {"accuracy": offline},
                    "representation_diagnostics": {"effective_rank": 90.0},
                },
                "online": {
                    "metrics": {"accuracy": online},
                    "representation_diagnostics": {"effective_rank": 90.5},
                },
            },
        }
        (tmp_path / f"seed-{seed}.json").write_text(json.dumps(record))

    analysis = analyze_online_continuation(tmp_path, expected_seeds=(11, 22))

    assert analysis["offline_accuracy"]["mean"] == pytest.approx(0.965)
    assert analysis["online_accuracy"]["mean"] == pytest.approx(0.965)
    assert analysis["accuracy_difference"]["mean"] == pytest.approx(0.0)
    assert analysis["effective_rank_difference"]["mean"] == pytest.approx(0.5)
    assert analysis["parameter_delta_l2"]["mean"] == pytest.approx(0.05)
