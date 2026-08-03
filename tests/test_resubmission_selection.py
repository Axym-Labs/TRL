import pytest

from terel.resubmission.selection import (
    evaluate_fidelity,
    load_selection_plan,
    validate_plan_protocol,
)


def test_selection_plan_is_bounded_and_configuration_ids_are_unique(tmp_path):
    plan = tmp_path / "selection.yaml"
    configurations = "\n".join(
        f"      - id: c{index}\n        learning_rate: 0.0003"
        for index in range(1, 13)
    )
    plan.write_text(
        f"""
seeds: [101, 202, 303]
probe:
  epochs: 30
datasets:
  mnist:
    configurations:
{configurations}
"""
    )

    loaded = load_selection_plan(plan)
    assert len(loaded["datasets"]["mnist"]["configurations"]) == 12

    plan.write_text(plan.read_text() + "      - id: c13\n        learning_rate: 0.001\n")
    with pytest.raises(ValueError, match="maximum of 12"):
        load_selection_plan(plan)


def test_selection_refuses_a_protocol_different_from_the_predeclared_hash(tmp_path):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("version one\n")

    with pytest.raises(ValueError, match="protocol hash"):
        validate_plan_protocol({"protocol_sha256": "not-the-hash"}, protocol)


def test_selection_fidelity_combines_scale_activity_and_effective_rank():
    results = [
        {
            "representation_diagnostics": {
                "mean_feature_variance": 0.01,
                "active_feature_fraction": 0.9,
                "effective_rank": 12.0,
            }
        },
        {
            "representation_diagnostics": {
                "mean_feature_variance": 0.02,
                "active_feature_fraction": 0.8,
                "effective_rank": 10.0,
            }
        },
    ]
    thresholds = {
        "mean_feature_variance": 0.001,
        "active_feature_fraction": 0.25,
        "effective_rank": 5.0,
    }

    valid, by_seed = evaluate_fidelity(results, seeds=(101, 202), thresholds=thresholds)

    assert valid is True
    assert by_seed["101"]["effective_rank"] == 12.0
    results[1]["representation_diagnostics"]["effective_rank"] = 4.9
    valid, _ = evaluate_fidelity(results, seeds=(101, 202), thresholds=thresholds)
    assert valid is False
