import pytest

from terel.resubmission.canonical_online_analysis import analyze_canonical_online


def _record(seed, accuracy, rank, *, base=(0.6, 0.7), state=(0.5, 0.4)):
    return {
        "seed": seed,
        "method": "terel_residual",
        "metrics": {"accuracy": accuracy, "macro_f1": accuracy - 0.001},
        "representation_diagnostics": {"effective_rank": rank},
        "encoder_training": {
            "base_residual_state_rms_mean": list(base),
            "residual_state_rms_mean": list(state),
            "optimizer_steps": 100,
            "causal_dynamic_state_numel": 8,
            "auxiliary_parameter_numel": 12,
            "parameter_numel": 10,
        },
        "resource_accounting": {"optimizer_state_bytes": 0},
    }


def test_canonical_analysis_separates_final_and_matched_validation_evidence():
    final = [_record(seed, 0.95 + seed / 10000, 140 + seed / 10) for seed in range(1, 6)]
    inhibited = [_record(seed, value, 145 + seed) for seed, value in zip((101, 102, 103), (0.96, 0.95, 0.955), strict=True)]
    reference = [_record(seed, value, 140 + seed) for seed, value in zip((101, 102, 103), (0.94, 0.93, 0.935), strict=True)]

    analysis = analyze_canonical_online(final, inhibited, reference)

    assert analysis["final"]["accuracy"]["mean"] == pytest.approx(0.9503)
    assert analysis["validation"]["paired_accuracy_gain"]["mean"] == pytest.approx(0.02)
    assert analysis["final"]["causal_dynamic_state_numel"] == 8
    assert analysis["final"]["auxiliary_parameter_numel"] == 12
    assert analysis["final"]["optimizer_state_bytes"] == 0


def test_canonical_analysis_requires_matched_validation_seeds():
    final = [_record(seed, 0.95, 140) for seed in range(1, 6)]
    inhibited = [_record(seed, 0.95, 145) for seed in (101, 102, 103)]
    reference = [_record(seed, 0.94, 140) for seed in (101, 102, 104)]

    with pytest.raises(ValueError, match="matched seeds"):
        analyze_canonical_online(final, inhibited, reference)
