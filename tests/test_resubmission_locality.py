import torch

from terel.resubmission.data import TemporalTensorDataset
from terel.resubmission.experiments import EncoderExperimentConfig
from terel.resubmission.locality import run_locality_audit


def test_locality_audit_compares_streaming_detached_and_batched_graph_scopes():
    dataset = TemporalTensorDataset(
        features=torch.randn(12, 3),
        labels=torch.tensor([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]),
        boundaries=torch.tensor(
            [True, False, False, False, False, False, True, False, False, False, False, False]
        ),
    )
    encoder = EncoderExperimentConfig(
        method="terel_local",
        hidden_dims=(4, 3),
        activation="identity",
        epochs=1,
        batch_size=4,
        order_mode="chronological",
        optimizer="adamw",
        learning_rate=1e-3,
        weight_decay=0.0,
        statistics_momentum=0.9,
        lateral_momentum=0.99,
    )

    audit = run_locality_audit(dataset=dataset, encoder=encoder, seed=9, device=torch.device("cpu"))

    assert set(audit["variants"]) == {
        "detached-stream-b1",
        "detached-minibatch",
        "undetached-minibatch",
    }
    streaming = audit["variants"]["detached-stream-b1"]
    batched = audit["variants"]["detached-minibatch"]
    assert streaming["batch_size"] == 1
    assert streaming["temporal_reference_detached"] is True
    assert audit["variants"]["undetached-minibatch"]["temporal_reference_detached"] is False
    assert streaming["training"]["steps"] == len(dataset)
    assert batched["training"]["steps"] == 3
    assert streaming["dynamic_state_numel_before"] == streaming["dynamic_state_numel_after"]
    assert streaming["state_retains_autograd_graph"] is False
    assert all(delta > 0 for delta in streaming["training"]["layer_parameter_delta_l2"])
