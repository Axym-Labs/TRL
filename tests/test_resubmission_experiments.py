import torch

from terel.resubmission import experiments
from terel.resubmission.data import DatasetSplits, TemporalTensorDataset
from terel.resubmission.experiments import (
    EncoderExperimentConfig,
    ProbeExperimentConfig,
    run_representation_experiment,
)
from terel.resubmission.provenance import TestGateError


def _toy_splits():
    features = torch.tensor(
        [
            [-2.0, -1.0],
            [-1.0, -2.0],
            [-2.0, -2.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [-1.5, -1.5],
            [1.5, 1.5],
        ]
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 0, 1])

    def dataset(index):
        return TemporalTensorDataset(
            features=features[index],
            labels=labels[index],
            boundaries=torch.tensor([True] + [False] * (len(index) - 1)),
        )

    return DatasetSplits(
        train=dataset(torch.tensor([0, 1, 3, 4, 6, 7])),
        validation=dataset(torch.tensor([2, 5])),
        test=dataset(torch.tensor([2, 5])),
        metadata={"dataset": "toy"},
    )


def _probe_config():
    return ProbeExperimentConfig(
        epochs=40,
        batch_size=3,
        optimizer="sgd",
        learning_rate=0.1,
        weight_decay=0.0,
    )


def test_matched_all_layer_readout_is_a_frozen_probe_choice():
    result = run_representation_experiment(
        splits=_toy_splits(),
        dataset_name="toy",
        num_classes=2,
        seed=101,
        encoder=EncoderExperimentConfig(
            method="random",
            hidden_dims=(4, 2),
            epochs=1,
            batch_size=3,
            order_mode="chronological",
        ),
        probe=ProbeExperimentConfig(
            epochs=40,
            batch_size=3,
            optimizer="sgd",
            learning_rate=0.1,
            weight_decay=0.0,
            readout="all",
        ),
        evaluation_split="validation",
        device=torch.device("cpu"),
    )

    assert result["probe_config"]["readout"] == "all"
    assert result["resource_accounting"]["probe_input_dim"] == 6


def test_invalid_probe_readout_is_rejected_before_training():
    try:
        run_representation_experiment(
            splits=_toy_splits(),
            dataset_name="toy",
            num_classes=2,
            seed=101,
            encoder=EncoderExperimentConfig(
                method="random",
                hidden_dims=(4, 2),
                epochs=1,
                batch_size=3,
            ),
            probe=ProbeExperimentConfig(readout="secret_labels"),
            evaluation_split="validation",
            device=torch.device("cpu"),
        )
    except ValueError as error:
        assert "readout" in str(error)
    else:
        raise AssertionError("invalid readout was accepted")


def test_default_statistics_rate_is_the_validated_noncollapse_setting():
    """A 0.99 default delays the variance gate long enough for ReLU collapse."""
    config = EncoderExperimentConfig(
        method="terel_local",
        hidden_dims=(4, 2),
    )

    assert config.statistics_momentum == 0.9


def test_seed_setup_enables_deterministic_torch_execution():
    from terel.resubmission.experiments import set_reproducible_seed

    set_reproducible_seed(101)
    first = torch.randn(8)
    set_reproducible_seed(101)
    second = torch.randn(8)

    assert torch.are_deterministic_algorithms_enabled()
    assert torch.equal(first, second)


def test_random_encoder_experiment_has_matched_probe_and_serializable_audit():
    result = run_representation_experiment(
        splits=_toy_splits(),
        dataset_name="toy",
        num_classes=2,
        seed=101,
        encoder=EncoderExperimentConfig(
            method="random",
            hidden_dims=(4, 2),
            epochs=1,
            batch_size=3,
            order_mode="chronological",
        ),
        probe=_probe_config(),
        evaluation_split="validation",
        device=torch.device("cpu"),
    )

    assert result["method"] == "random"
    assert result["evaluation_split"] == "validation"
    assert result["encoder_training"] is None
    assert result["probe_training"]["examples"] == 40 * 6
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0
    assert result["resource_accounting"]["parameter_bytes"] > 0
    assert result["class_structure_diagnostics"]["nearest_centroid_accuracy"] >= 0.5


def test_calibrated_random_encoder_serializes_normalization_separately():
    result = run_representation_experiment(
        splits=_toy_splits(),
        dataset_name="toy",
        num_classes=2,
        seed=101,
        encoder=EncoderExperimentConfig(
            method="random",
            hidden_dims=(4, 2),
            normalization="batch_norm",
            batch_norm_calibration_passes=1,
            epochs=1,
            batch_size=3,
            order_mode="chronological",
        ),
        probe=_probe_config(),
        evaluation_split="validation",
        device=torch.device("cpu"),
    )

    assert result["encoder_training"] is None
    assert result["normalization_calibration"]["passes"] == 1
    assert result["normalization_calibration"]["batches"] == 2
    assert result["normalization_calibration"]["examples"] == 6
    assert result["resource_accounting"]["normalization_calibration_examples"] == 6


def test_corrected_terel_experiment_records_all_layer_updates():
    result = run_representation_experiment(
        splits=_toy_splits(),
        dataset_name="toy",
        num_classes=2,
        seed=101,
        encoder=EncoderExperimentConfig(
            method="terel_local",
            hidden_dims=(4, 2),
            activation="identity",
            epochs=2,
            batch_size=3,
            order_mode="chronological",
            optimizer="sgd",
            learning_rate=0.01,
            statistics_momentum=0.9,
            lateral_momentum=0.9,
        ),
        probe=_probe_config(),
        evaluation_split="validation",
        device=torch.device("cpu"),
    )

    assert all(delta > 0.0 for delta in result["encoder_training"]["layer_parameter_delta_l2"])
    assert all(delta > 0.0 for delta in result["encoder_training"]["layer_lateral_delta_l2"])
    assert result["encoder_training"]["examples"] == 12
    assert result["resource_accounting"]["operation_proxy"]["linear_forward_backward_mac_proxy"] > 0
    assert result["resource_accounting"]["operation_proxy"]["same_layer_pairwise_mac_proxy"] > 0
    assert "temporal_slowness" in result["representation_diagnostics"]


def test_residual_state_experiment_runs_through_the_cpu_validation_path():
    """Omitting the new mode from configuration or dispatch must break this run."""
    result = run_representation_experiment(
        splits=_toy_splits(),
        dataset_name="toy",
        num_classes=2,
        seed=101,
        encoder=EncoderExperimentConfig(
            method="terel_residual",
            hidden_dims=(4, 2),
            activation="identity",
            normalization="streaming_norm",
            normalization_affine=False,
            epochs=1,
            batch_size=1,
            order_mode="chronological",
            optimizer="sgd",
            learning_rate=0.01,
            weight_decay=0.0,
            statistics_momentum=0.9,
            lateral_momentum=0.9,
            residual_lateral_steps=2,
            residual_lateral_step_size=0.1,
            residual_lateral_rule="dual_inhibitory",
            residual_lateral_coefficient=2.0,
        ),
        probe=_probe_config(),
        evaluation_split="validation",
        device=torch.device("cpu"),
    )

    assert result["method"] == "terel_residual"
    assert result["encoder_config"]["residual_lateral_steps"] == 2
    assert result["encoder_config"]["normalization_affine"] is False
    assert result["encoder_config"]["residual_lateral_rule"] == "dual_inhibitory"
    assert all(delta > 0.0 for delta in result["encoder_training"]["layer_parameter_delta_l2"])
    assert all(delta > 0.0 for delta in result["encoder_training"]["layer_lateral_delta_l2"])
    assert all(
        delta > 0.0
        for delta in result["encoder_training"]["residual_lateral_delta_l2"]
    )
    assert all(value > 0.0 for value in result["encoder_training"]["residual_state_rms_mean"])
    assert all(
        value >= 0.0
        for value in result["encoder_training"]["residual_dynamics_delta_rms_mean"]
    )
    assert result["resource_accounting"]["encoder_batch_size"] == 1


def test_greedy_training_budget_is_recorded_as_epochs_per_layer():
    result = run_representation_experiment(
        splits=_toy_splits(),
        dataset_name="toy",
        num_classes=2,
        seed=101,
        encoder=EncoderExperimentConfig(
            method="terel_batch",
            hidden_dims=(4, 2),
            activation="identity",
            epochs=2,
            batch_size=3,
            order_mode="chronological",
            optimizer="sgd",
            learning_rate=0.01,
            weight_decay=0.0,
            statistics_momentum=0.9,
            lateral_momentum=0.9,
            training_mode="greedy",
        ),
        probe=_probe_config(),
        evaluation_split="validation",
        device=torch.device("cpu"),
    )

    assert result["encoder_training"]["training_mode"] == "greedy"
    assert result["encoder_training"]["epochs_per_layer"] == 2
    assert result["encoder_training"]["examples"] == 2 * 2 * 6
    assert all(delta > 0.0 for delta in result["encoder_training"]["layer_parameter_delta_l2"])
    proxy = result["resource_accounting"]["operation_proxy"]
    assert proxy["linear_forward_backward_mac_proxy"] == 672
    assert proxy["same_layer_pairwise_mac_proxy"] == 480


def test_undetached_direct_control_keeps_the_canonical_temporal_gradient():
    """The matched direct control must not silently detach the preceding activation."""
    assert hasattr(experiments, "resolve_terel_objective_mode")
    detach_previous, covariance_mode = experiments.resolve_terel_objective_mode(
        "terel_direct_batch"
    )

    assert detach_previous is False
    assert covariance_mode == "direct"


def test_held_out_split_cannot_run_without_gate_context():
    try:
        run_representation_experiment(
            splits=_toy_splits(),
            dataset_name="toy",
            num_classes=2,
            seed=1001,
            encoder=EncoderExperimentConfig(
                method="random",
                hidden_dims=(2,),
                epochs=1,
                batch_size=2,
                order_mode="chronological",
            ),
            probe=_probe_config(),
            evaluation_split="test",
            device=torch.device("cpu"),
        )
    except TestGateError as error:
        assert "gate context" in str(error)
    else:
        raise AssertionError("held-out test evaluation ran without a gate")


def test_incremental_sfa_experiment_reports_streaming_pair_budget():
    result = run_representation_experiment(
        splits=_toy_splits(),
        dataset_name="toy",
        num_classes=2,
        seed=101,
        encoder=EncoderExperimentConfig(
            method="incsfa",
            hidden_dims=(2,),
            epochs=2,
            batch_size=1,
            order_mode="chronological",
            incsfa_whitening_dim=2,
            incsfa_output_dim=1,
            incsfa_learning_rate=0.05,
        ),
        probe=_probe_config(),
        evaluation_split="validation",
        device=torch.device("cpu"),
    )

    assert result["encoder_training"]["examples"] == 12
    assert result["encoder_training"]["valid_temporal_pairs"] == 10
    assert result["resource_accounting"]["dynamic_state_bytes"] > 0


def test_batch_sfa_experiment_reports_its_single_pass_budget():
    result = run_representation_experiment(
        splits=_toy_splits(),
        dataset_name="toy",
        num_classes=2,
        seed=101,
        encoder=EncoderExperimentConfig(
            method="sfa",
            hidden_dims=(1,),
            epochs=10,
            batch_size=6,
            order_mode="chronological",
            sfa_components=1,
        ),
        probe=_probe_config(),
        evaluation_split="validation",
        device=torch.device("cpu"),
    )

    assert result["encoder_training"]["epochs"] == 1
    assert result["encoder_training"]["steps"] == 1
    assert result["encoder_training"]["examples"] == 6
    assert result["encoder_training"]["valid_temporal_pairs"] == 5
    assert result["resource_accounting"]["operation_proxy"]["training_examples"] == 6
