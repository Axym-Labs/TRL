from terel.resubmission.residual_state_analysis import (
    analyze_residual_state_evaluation,
    render_appendix_latex,
    render_results_latex,
)


def _record(seed, accuracy, rank, base=(0.5, 0.6), settled=(0.4, 0.3)):
    return {
        "seed": seed,
        "evaluation_split": "test",
        "method": "terel_residual",
        "metrics": {"accuracy": accuracy},
        "representation_diagnostics": {
            "effective_rank": rank,
            "median_feature_variance": 0.3,
        },
        "encoder_training": {
            "base_residual_state_rms_mean": list(base),
            "residual_state_rms_mean": list(settled),
            "seconds": 10.0 + seed,
            "optimizer_steps": 100,
            "dynamic_state_numel": 20,
        },
        "resource_accounting": {
            "parameter_bytes": 40,
            "optimizer_state_bytes": 80,
            "dynamic_state_bytes": 100,
        },
    }


def test_analysis_summarizes_frozen_final_and_validation_effects():
    records = [
        _record(seed, accuracy, rank)
        for seed, accuracy, rank in zip(
            (1, 2, 3, 4, 5),
            (0.94, 0.95, 0.96, 0.95, 0.95),
            (80.0, 81.0, 82.0, 81.0, 81.0),
            strict=True,
        )
    ]
    validation = {
        "selection_complete": True,
        "paired_accuracy": {
            "terel_s_reference": {"values": [0.90, 0.91, 0.92]},
            "terel_s_residual": {"values": [0.94, 0.95, 0.96]},
            "residual_minus_reference": {"values": [0.04, 0.04, 0.04]},
        },
        "effective_rank": {
            "terel_s_reference": {"values": [79.0, 80.0, 81.0]},
            "terel_s_residual": {"values": [82.0, 83.0, 84.0]},
        },
    }

    analysis = analyze_residual_state_evaluation(records, validation)

    assert analysis["final"]["accuracy"]["mean"] == 0.95
    assert analysis["final"]["effective_rank"]["mean"] == 81.0
    assert analysis["final"]["base_state_rms"][0]["mean"] == 0.5
    assert analysis["final"]["settled_state_rms"][1]["mean"] == 0.3
    assert analysis["validation"]["paired_accuracy_gain"]["mean"] == 0.04


def test_renderers_keep_final_and_validation_roles_distinct():
    records = [_record(seed, 0.95, 81.0) for seed in (1, 2, 3, 4, 5)]
    validation = {
        "selection_complete": True,
        "paired_accuracy": {
            "terel_s_reference": {"values": [0.91, 0.91, 0.91]},
            "terel_s_residual": {"values": [0.95, 0.95, 0.95]},
            "residual_minus_reference": {"values": [0.04, 0.04, 0.04]},
        },
        "effective_rank": {
            "terel_s_reference": {"values": [80.0, 80.0, 80.0]},
            "terel_s_residual": {"values": [83.0, 83.0, 83.0]},
        },
    }
    analysis = analyze_residual_state_evaluation(records, validation)

    main = render_results_latex(analysis)
    appendix = render_appendix_latex(analysis)

    assert "Frozen final evaluation" in main
    assert "Matched validation effect" in main
    assert "95.00" in main
    assert "4.00" in main
    assert "Raw final-run values" in appendix
    assert "1, 2, 3, 4, 5" in appendix
    assert all(line == line.rstrip() for line in appendix.splitlines())


def test_analysis_rejects_nonfinal_or_duplicate_records():
    records = [_record(seed, 0.95, 81.0) for seed in (1, 2, 3, 4, 4)]
    validation = {"selection_complete": True}

    try:
        analyze_residual_state_evaluation(records, validation)
    except ValueError as error:
        assert "distinct" in str(error)
    else:
        raise AssertionError("duplicate final records were accepted")
