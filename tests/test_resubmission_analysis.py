import json

import numpy as np

from terel.resubmission.analysis import analyze_confirmatory_results


def _write_result(root, dataset, run_id, seed, metric_name, value):
    path = root / dataset / run_id / f"seed-{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "run_id": run_id,
                "seed": seed,
                "metrics": {metric_name: value},
                "representation_diagnostics": {"median_feature_variance": 0.2},
                "class_structure_diagnostics": {"nearest_centroid_accuracy": 0.7},
                "resource_accounting": {"parameter_bytes": 100},
            }
        )
    )


def test_confirmatory_analysis_reports_seed_summaries_and_paired_primary_effects(tmp_path):
    for seed, treatment, control in [(1, 0.8, 0.5), (2, 0.9, 0.6)]:
        _write_result(tmp_path, "mnist", "terel-local", seed, "accuracy", treatment)
        _write_result(tmp_path, "mnist", "random", seed, "accuracy", control)
    for seed, ordered, shuffled in [(1, 0.6, 0.55), (2, 0.7, 0.65)]:
        _write_result(tmp_path, "pamap2", "terel-ordered", seed, "macro_f1", ordered)
        _write_result(tmp_path, "pamap2", "terel-shuffled", seed, "macro_f1", shuffled)

    analysis = analyze_confirmatory_results(
        tmp_path,
        expected_seeds=(1, 2),
        bootstrap_samples=100,
        bootstrap_seed=17,
    )

    assert np.isclose(analysis["datasets"]["mnist"]["methods"]["terel-local"]["mean"], 0.85)
    assert np.isclose(
        analysis["primary_contrasts"]["mnist_terel_minus_random"]["mean_difference"],
        0.3,
    )
    assert np.isclose(
        analysis["primary_contrasts"]["pamap2_ordered_minus_shuffled"]["mean_difference"],
        0.05,
    )
