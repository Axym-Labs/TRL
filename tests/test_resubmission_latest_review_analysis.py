import json

import pytest


def _write_record(root, run_id, seed, accuracy, *, normalization, calibrated=False):
    directory = root / "mnist" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": "mnist",
        "run_id": run_id,
        "seed": seed,
        "metrics": {"accuracy": accuracy},
        "encoder_config": {"normalization": normalization},
        "encoder_training": None,
        "normalization_calibration": (
            {"passes": 1, "batches": 196, "examples": 50_000, "seconds": 1.0}
            if calibrated
            else None
        ),
    }
    (directory / f"seed-{seed}.json").write_text(json.dumps(record))


def test_latest_review_analysis_reports_matched_random_control_and_paired_effects(tmp_path):
    from terel.resubmission.latest_review_analysis import (
        analyze_normalization_control,
        render_normalization_control_latex,
    )

    control_root = tmp_path / "control"
    reference_root = tmp_path / "reference"
    for seed, terel, calibrated, unnormalized in (
        (11, 0.97, 0.94, 0.90),
        (22, 0.98, 0.95, 0.91),
    ):
        _write_record(
            control_root,
            "random-bn-calibrated-all",
            seed,
            calibrated,
            normalization="batch_norm",
            calibrated=True,
        )
        _write_record(reference_root, "terel-all", seed, terel, normalization="batch_norm")
        _write_record(reference_root, "random-all", seed, unnormalized, normalization="none")

    analysis = analyze_normalization_control(
        control_root,
        reference_root,
        seeds=(11, 22),
    )

    assert analysis["random_bn_calibrated"]["mean"] == pytest.approx(0.945)
    assert analysis["terel_minus_random_bn"]["mean_difference"] == pytest.approx(0.03)
    assert analysis["random_bn_minus_random_no_norm"]["mean_difference"] == pytest.approx(0.04)
    assert analysis["decision"] == "supported"
    rendered = render_normalization_control_latex(analysis)
    assert "BatchNorm-calibrated random" in rendered
    assert "3.00" in rendered


def test_latest_review_analysis_rejects_an_uncalibrated_or_trained_control(tmp_path):
    from terel.resubmission.latest_review_analysis import analyze_normalization_control

    control_root = tmp_path / "control"
    reference_root = tmp_path / "reference"
    _write_record(
        control_root,
        "random-bn-calibrated-all",
        11,
        0.94,
        normalization="batch_norm",
        calibrated=False,
    )
    _write_record(reference_root, "terel-all", 11, 0.97, normalization="batch_norm")
    _write_record(reference_root, "random-all", 11, 0.90, normalization="none")

    with pytest.raises(ValueError, match="calibration"):
        analyze_normalization_control(control_root, reference_root, seeds=(11,))
