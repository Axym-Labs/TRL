import torch

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


def test_default_statistics_rate_is_the_validated_noncollapse_setting():
    """A 0.99 default delays the variance gate long enough for ReLU collapse."""
    config = EncoderExperimentConfig(
        method="terel_local",
        hidden_dims=(4, 2),
    )

    assert config.statistics_momentum == 0.9


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
