import json

import pytest

from terel.resubmission.mechanism_analysis_v2 import analyze_mechanism_audit


def _write(root, candidate, seed, accuracy, rank=20.0, variance=0.2):
    directory = root / candidate
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"seed-{seed}.json").write_text(
        json.dumps(
            {
                "metrics": {"accuracy": accuracy},
                "representation_diagnostics": {
                    "effective_rank": rank,
                    "median_feature_variance": variance,
                },
                "class_structure_diagnostics": {"nearest_centroid_accuracy": accuracy},
            }
        )
    )


def test_mechanism_analysis_pairs_each_intervention_with_full(tmp_path):
    reference = tmp_path / "reference"
    audit = tmp_path / "audit"
    for seed, accuracy in ((11, 0.97), (22, 0.98)):
        _write(reference, "canonical-recovered-bn", seed, accuracy)
        for candidate in ("no-temporal", "no-variance", "no-covariance", "shuffled-order"):
            _write(audit, candidate, seed, accuracy - 0.10)

    result = analyze_mechanism_audit(reference, audit, seeds=(11, 22))

    assert result["methods"]["full"]["accuracy"]["mean"] == pytest.approx(0.975)
    assert result["methods"]["no-temporal"]["accuracy_difference_from_full"][
        "mean_difference"
    ] == pytest.approx(-0.10)
