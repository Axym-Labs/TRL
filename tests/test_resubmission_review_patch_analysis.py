import importlib
import importlib.util
import json

import pytest



def _analysis_module():
    assert importlib.util.find_spec("terel.resubmission.review_patch_analysis") is not None
    return importlib.import_module("terel.resubmission.review_patch_analysis")


def _write_record(root, candidate, seed, accuracy, *, alignment=None):
    path = root / candidate / f"seed-{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    training = {"seconds": 1.0, "examples": 100, "optimizer_steps": 10}
    if alignment is not None:
        training.update(
            {
                "lateral_proxy_cosine_mean": list(alignment),
                "lateral_proxy_relative_error_mean": [0.2, 0.3],
                "lateral_proxy_norm_ratio_mean": [0.9, 1.1],
                "lateral_proxy_audited_batches": [9, 9],
            }
        )
    path.write_text(
        json.dumps(
            {
                "candidate_id": candidate,
                "seed": seed,
                "metrics": {"accuracy": accuracy},
                "encoder_training": training,
                "encoder_config": {"learning_rate": 0.001, "contrastive_temperature": 0.1},
            }
        )
    )


def test_review_patch_analysis_selects_validation_winner_and_summarizes_controls(
    tmp_path,
):
    review_patch_analysis = _analysis_module()
    for seed, value in zip((1, 2, 3), (0.91, 0.92, 0.93), strict=True):
        _write_record(tmp_path, "candidate-a", seed, value)
    for seed, value in zip((1, 2, 3), (0.94, 0.95, 0.96), strict=True):
        _write_record(tmp_path, "candidate-b", seed, value)
    for seed, value in zip((1, 2, 3), (0.97, 0.98, 0.99), strict=True):
        _write_record(tmp_path, "lagged", seed, value, alignment=(0.8, 0.6))
    for seed, value in zip((1, 2, 3), (0.96, 0.97, 0.98), strict=True):
        _write_record(tmp_path, "direct", seed, value)

    analysis = review_patch_analysis.analyze_validation_patch(
        tmp_path,
        local_candidates=("candidate-a", "candidate-b"),
        validation_seeds=(1, 2, 3),
        lagged_id="lagged",
        direct_id="direct",
    )

    assert analysis["selected_local_supcon"] == "candidate-b"
    assert analysis["local_supcon_candidates"]["candidate-b"]["mean"] == pytest.approx(
        0.95
    )
    assert analysis["lateral_controls"]["lagged_proxy"]["cosine_alignment_by_layer"] == [
        pytest.approx(0.8),
        pytest.approx(0.6),
    ]
    assert analysis["lateral_controls"]["direct_minus_lagged"][
        "mean_difference"
    ] == pytest.approx(-0.01)

    ledger = review_patch_analysis.build_review_patch_validation_ledger(
        tmp_path,
        analysis,
        local_candidates=("candidate-a", "candidate-b"),
        validation_seeds=(1, 2, 3),
        lagged_id="lagged",
        direct_id="direct",
    )
    assert ledger["selection_complete"] is True
    assert ledger["selected_primary"] == "candidate-b"
    assert ledger["records"]["candidate-b"]["values"] == [0.94, 0.95, 0.96]
    assert len(ledger["records"]["candidate-b"]["sha256"]) == 3


def test_review_patch_analysis_refuses_an_incomplete_candidate(tmp_path):
    review_patch_analysis = _analysis_module()
    _write_record(tmp_path, "candidate-a", 1, 0.91)

    with pytest.raises(ValueError, match="missing validation record"):
        review_patch_analysis.analyze_validation_patch(
            tmp_path,
            local_candidates=("candidate-a",),
            validation_seeds=(1, 2),
            lagged_id="lagged",
            direct_id="direct",
        )
