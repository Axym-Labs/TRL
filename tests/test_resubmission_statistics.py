import numpy as np

from terel.resubmission.statistics import paired_contrast, summarize_values


def test_seed_summary_reports_sample_sd_and_reproducible_percentile_interval():
    summary = summarize_values([0.7, 0.8, 0.9, 1.0, 1.1], bootstrap_samples=10_000, seed=260803)

    assert np.isclose(summary["mean"], 0.9)
    assert np.isclose(summary["sample_sd"], np.std([0.7, 0.8, 0.9, 1.0, 1.1], ddof=1))
    assert summary["raw"] == [0.7, 0.8, 0.9, 1.0, 1.1]
    assert summary["bootstrap_samples"] == 10_000
    assert summary["ci95_low"] < summary["mean"] < summary["ci95_high"]


def test_paired_contrast_matches_by_seed_instead_of_result_order():
    treatment = {1001: 0.80, 1002: 0.75, 1003: 0.90, 1004: 0.85, 1005: 0.70}
    control = {1005: 0.60, 1003: 0.75, 1001: 0.70, 1004: 0.80, 1002: 0.70}

    contrast = paired_contrast(treatment, control, bootstrap_samples=10_000, seed=260803)

    assert contrast["seeds"] == [1001, 1002, 1003, 1004, 1005]
    assert np.allclose(contrast["raw_differences"], [0.10, 0.05, 0.15, 0.05, 0.10])
    assert np.isclose(contrast["mean_difference"], 0.09)
