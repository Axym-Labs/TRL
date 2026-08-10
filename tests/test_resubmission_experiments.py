import pytest
import torch

from terel.resubmission import experiments
from terel.resubmission.data import DatasetSplits, TemporalTensorDataset
from terel.resubmission.experiments import (
    EncoderExperimentConfig,
    ProbeExperimentConfig,
    run_representation_experiment,
)


def test_plain_sgd_has_no_momentum_or_adaptive_state():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = experiments._optimizer(
        "plain_sgd",
        [parameter],
        learning_rate=0.1,
        weight_decay=0.0,
    )

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["momentum"] == 0.0
    parameter.grad = torch.tensor([2.0])
    optimizer.step()
    assert optimizer.state == {}


def test_plain_sgd_weight_decay_is_the_declared_leaky_integrator_update():
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = experiments._optimizer(
        "plain_sgd",
        [parameter],
        learning_rate=0.1,
        weight_decay=0.2,
    )
    parameter.grad = torch.tensor([3.0])

    optimizer.step()

    assert torch.allclose(
        parameter, torch.tensor([(1.0 - 0.1 * 0.2) * 2.0 - 0.1 * 3.0])
    )
    dynamics = experiments._optimizer_dynamics(
        "plain_sgd", learning_rate=0.1, weight_decay=0.2
    )
    assert dynamics["exact_leaky_integrator"] is True
    assert dynamics["weight_retention_per_step"] == pytest.approx(0.98)


def test_residual_operation_proxy_counts_both_dense_operators():
    proxy = experiments._operation_proxy(
        "terel_residual",
        input_dim=2,
        hidden_dims=(3, 2),
        num_classes=2,
        batch_size=1,
        training={"examples": 10, "training_mode": "joint"},
    )

    assert proxy["same_layer_pairwise_mac_proxy"] == 4 * 10 * (3**2 + 2**2)

    four_pass_proxy = experiments._operation_proxy(
        "terel_residual",
        input_dim=2,
        hidden_dims=(3, 2),
        num_classes=2,
        batch_size=1,
        training={"examples": 10, "training_mode": "joint"},
        residual_lateral_steps=4,
    )

    assert four_pass_proxy["same_layer_pairwise_mac_proxy"] == 7 * 10 * (3**2 + 2**2)


def test_equal_offset_experiment_accounts_for_both_delayed_vectors():
    """An equal-offset run must expose its larger causal state rather than hiding it."""
    result = run_representation_experiment(
        splits=_toy_splits(),
        dataset_name="toy",
        num_classes=2,
        seed=101,
        encoder=EncoderExperimentConfig(
            method="terel_residual",
            hidden_dims=(4, 2),
            activation="identity",
            epochs=1,
            batch_size=1,
            order_mode="chronological",
            optimizer="plain_sgd",
            learning_rate=0.001,
            statistics_momentum=0.9,
            lateral_momentum=0.9,
            residual_lateral_coefficient=1.0,
            residual_lateral_steps=1,
            residual_lateral_signal_offset=1,
        ),
        probe=_probe_config(),
        evaluation_split="validation",
        device=torch.device("cpu"),
    )

    widths = 4 + 2
    flags = 2
    assert result["resource_accounting"]["causal_dynamic_state_bytes"] == (
        5 * widths * torch.tensor(0.0).element_size() + flags
    )
    assert result["encoder_config"]["residual_lateral_signal_offset"] == 1


@pytest.mark.parametrize(
    "matrix_mode", ["representation_shared", "state_shared", "combined"]
)
def test_one_matrix_residual_candidates_report_one_auxiliary_matrix(matrix_mode):
    result = run_representation_experiment(
        splits=_toy_splits(),
        dataset_name="toy",
        num_classes=2,
        seed=103,
        encoder=EncoderExperimentConfig(
            method="terel_residual",
            hidden_dims=(4, 2),
            activation="relu",
            epochs=1,
            batch_size=1,
            order_mode="chronological",
            optimizer="plain_sgd",
            learning_rate=0.001,
            lateral_matrix_mode=matrix_mode,
            combined_lateral_state_weight=0.5,
        ),
        probe=_probe_config(),
        evaluation_split="validation",
        device=torch.device("cpu"),
    )

    expected = (4**2 + 2**2) * torch.tensor(0.0).element_size()
    assert result["resource_accounting"]["auxiliary_parameter_bytes"] == expected


def test_adam_component_ablation_accepts_explicit_betas():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = experiments._optimizer(
        "adamw",
        [parameter],
        learning_rate=0.1,
        weight_decay=0.0,
        beta1=0.0,
        beta2=0.99,
        epsilon=1e-6,
    )

    assert optimizer.param_groups[0]["betas"] == (0.0, 0.99)
    assert optimizer.param_groups[0]["eps"] == 1e-6


def test_online_inference_continues_label_free_terel_after_probe_fitting():
    result = run_representation_experiment(
        splits=_toy_splits(),
        dataset_name="toy",
        num_classes=2,
        seed=13,
        encoder=EncoderExperimentConfig(
            method="terel_local",
            hidden_dims=(3,),
            activation="identity",
            epochs=1,
            batch_size=1,
            order_mode="chronological",
            optimizer="plain_sgd",
            learning_rate=0.001,
            inference_mode="online",
        ),
        probe=_probe_config(),
        evaluation_split="validation",
        device=torch.device("cpu"),
    )

    assert result["online_inference"]["examples"] == len(_toy_splits().validation)
    assert result["online_inference"]["labels_accessed"] is False
    assert result["online_inference"]["optimizer_steps"] == len(
        _toy_splits().validation
    )


def test_supervised_linear_reference_uses_raw_inputs_without_encoder_training():
    result = run_representation_experiment(
        splits=_toy_splits(),
        dataset_name="toy",
        num_classes=2,
        seed=19,
        encoder=EncoderExperimentConfig(
            method="supervised_linear",
            hidden_dims=(99,),
        ),
        probe=_probe_config(),
        evaluation_split="validation",
        device=torch.device("cpu"),
    )

    assert result["encoder_training"] is None
    assert result["resource_accounting"]["probe_input_dim"] == 2
    assert result["resource_accounting"]["dynamic_state_bytes"] == 0
    assert result["metrics"]["accuracy"] >= 0.5


def test_terel_offline_is_end_to_end_final_layer_soft_sfa_without_normalization():
    result = run_representation_experiment(
        splits=_toy_splits(),
        dataset_name="toy",
        num_classes=2,
        seed=23,
        encoder=EncoderExperimentConfig(
            method="terel_offline",
            hidden_dims=(4, 2),
            activation="relu",
            normalization="none",
            epochs=2,
            batch_size=3,
            order_mode="chronological",
            optimizer="plain_sgd",
            learning_rate=0.01,
            weight_decay=0.001,
        ),
        probe=_probe_config(),
        evaluation_split="validation",
        device=torch.device("cpu"),
    )

    assert result["encoder_training"]["training_mode"] == "end_to_end_subsequence"
    assert len(result["encoder_training"]["layer_parameter_delta_l2"]) == 2
    assert result["resource_accounting"]["dynamic_state_bytes"] == 0
    assert result["encoder_config"]["normalization"] == "none"


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


def test_representation_probe_excludes_unknown_labels_without_dropping_encoder_rows():
    """Using -1 labels in the probe or dropping their encoder observations is invalid."""
    splits = _toy_splits()
    splits.train.labels[1] = -1
    splits.train.labels[4] = -1
    splits.validation.features = torch.cat(
        (splits.validation.features[:1], splits.validation.features), dim=0
    )
    splits.validation.labels = torch.tensor([-1, 0, 1])
    splits.validation.boundaries = torch.tensor([True, False, False])

    result = run_representation_experiment(
        splits=splits,
        dataset_name="toy-masked-labels",
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

    assert result["probe_training"]["examples"] == _probe_config().epochs * 4
    assert result["resource_accounting"]["encoder_batch_size"] == 3
    assert result["metrics"]["support"] == 2


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

    assert all(
        delta > 0.0 for delta in result["encoder_training"]["layer_parameter_delta_l2"]
    )
    assert all(
        delta > 0.0 for delta in result["encoder_training"]["layer_lateral_delta_l2"]
    )
    assert result["encoder_training"]["examples"] == 12
    assert (
        result["resource_accounting"]["operation_proxy"][
            "linear_forward_backward_mac_proxy"
        ]
        > 0
    )
    assert (
        result["resource_accounting"]["operation_proxy"][
            "same_layer_pairwise_mac_proxy"
        ]
        > 0
    )
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
            audit_residual_components=True,
        ),
        probe=_probe_config(),
        evaluation_split="validation",
        device=torch.device("cpu"),
    )

    assert result["method"] == "terel_residual"
    assert result["resource_accounting"]["causal_dynamic_state_bytes"] > 0
    assert result["resource_accounting"]["auxiliary_parameter_bytes"] > 0
    assert (
        result["resource_accounting"]["causal_dynamic_state_bytes"]
        < result["resource_accounting"]["auxiliary_parameter_bytes"]
    )
    assert result["encoder_config"]["residual_lateral_steps"] == 2
    assert result["encoder_config"]["normalization_affine"] is False
    assert result["encoder_config"]["residual_lateral_rule"] == "dual_inhibitory"
    assert all(
        delta > 0.0 for delta in result["encoder_training"]["layer_parameter_delta_l2"]
    )
    assert all(
        delta > 0.0 for delta in result["encoder_training"]["layer_lateral_delta_l2"]
    )
    assert all(
        delta > 0.0 for delta in result["encoder_training"]["residual_lateral_delta_l2"]
    )
    assert all(
        value > 0.0 for value in result["encoder_training"]["residual_state_rms_mean"]
    )
    assert all(
        value >= 0.0
        for value in result["encoder_training"]["residual_dynamics_delta_rms_mean"]
    )
    for term in ("temporal", "variance", "covariance"):
        values = result["encoder_training"][f"{term}_state_rms_mean"]
        assert len(values) == 2
        assert all(value >= 0.0 for value in values)
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
    assert all(
        delta > 0.0 for delta in result["encoder_training"]["layer_parameter_delta_l2"]
    )
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
