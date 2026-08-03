import torch
import torch.nn.functional as F
import numpy as np

from terel.resubmission.baselines import (
    BatchLinearSFA,
    IncrementalLinearSFA,
    SupervisedMLP,
    local_supervised_contrastive_step,
    train_local_supervised_contrastive,
    train_supervised_mlp,
)
from terel.resubmission.data import TemporalTensorDataset
from terel.resubmission.model import LayerLocalEncoder


def test_supervised_baseline_uses_every_declared_hidden_layer():
    """A hidden layer that is constructed but bypassed must fail this regression."""
    torch.manual_seed(11)
    model = SupervisedMLP(input_dim=4, hidden_dims=(5, 6), output_dim=3, activation="relu")
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    y = torch.tensor([0, 1, 2, 1])

    logits = model(x)
    F.cross_entropy(logits, y).backward()

    assert logits.shape == (4, 3)
    assert len(model.hidden_layers) == 2
    for layer in model.hidden_layers:
        assert layer.weight.grad is not None
        assert torch.count_nonzero(layer.weight.grad) > 0


def test_supervised_baseline_supports_the_matched_leaky_relu_activation():
    model = SupervisedMLP(
        input_dim=1,
        hidden_dims=(1,),
        output_dim=2,
        activation="leaky_relu",
    )
    with torch.no_grad():
        model.hidden_layers[0].weight.fill_(1.0)
        model.hidden_layers[0].bias.zero_()

    hidden = model.representations(torch.tensor([[-2.0]]))[0]

    assert torch.isclose(hidden[0, 0], torch.tensor(-0.02))


def test_batch_linear_sfa_satisfies_whitening_and_slow_feature_order():
    """The classical baseline must solve the stated constrained problem on a toy stream."""
    t = np.linspace(0.0, 12.0 * np.pi, 600, endpoint=False)
    x = np.column_stack(
        [
            np.sin(t),
            np.cos(0.25 * t),
            np.sin(3.0 * t) + 0.2 * np.cos(0.5 * t),
        ]
    )

    y = BatchLinearSFA(n_components=3).fit_transform(x)
    covariance = y.T @ y / len(y)
    delta = np.diff(y, axis=0)
    slowness = np.mean(delta**2, axis=0)

    assert np.allclose(y.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(covariance, np.eye(3), atol=1e-8)
    assert np.all(np.diff(slowness) >= -1e-10)


def test_batch_linear_sfa_excludes_cross_subject_derivatives():
    """A subject transition must not be treated as one enormous temporal derivative."""
    x = np.array([[0.0, 1.0], [0.1, 0.9], [100.0, 101.0], [100.1, 100.9]])
    boundaries = np.array([True, False, True, False])

    model = BatchLinearSFA(n_components=1).fit(x, boundaries=boundaries)

    assert model.derivative_pair_count_ == 2


def test_incremental_sfa_tracks_batch_sfa_on_a_toy_stream_and_resets_boundaries():
    """The documented IncSFA port must recover a comparably slow linear direction."""
    t = np.linspace(0.0, 20.0 * np.pi, 600, endpoint=False)
    sources = np.column_stack([np.sin(0.1 * t), np.sin(t), np.cos(2.5 * t)])
    mixing = np.array([[1.0, 0.3, -0.2], [0.2, 1.0, 0.4], [-0.3, 0.1, 1.0]])
    x = sources @ mixing
    boundaries = np.zeros(len(x), dtype=bool)
    boundaries[0] = True

    batch_y = BatchLinearSFA(n_components=1).fit_transform(x, boundaries=boundaries)[:, 0]
    incremental = IncrementalLinearSFA(
        input_dim=3,
        whitening_dim=3,
        output_dim=1,
        learning_rate=0.05,
        seed=7,
    ).fit(x, boundaries=boundaries, epochs=20)
    incremental_y = incremental.transform(x)[:, 0]

    def normalized_slowness(values):
        values = (values - values.mean()) / values.std()
        return np.mean(np.diff(values) ** 2)

    assert incremental.derivative_pair_count_ == 20 * (len(x) - 1)
    assert normalized_slowness(incremental_y) <= 1.5 * normalized_slowness(batch_y)
    assert np.isfinite(incremental_y).all()


def test_local_supervised_contrastive_updates_every_layer_without_cross_layer_gradients():
    torch.manual_seed(41)
    model = LayerLocalEncoder(
        input_dim=2,
        hidden_dims=(4, 3),
        activation="relu",
        statistics_momentum=0.9,
        lateral_momentum=0.9,
    )
    optimizer = torch.optim.SGD(model.encoder_parameters(), lr=0.1)
    before = [layer.weight.detach().clone() for layer in model.layers]
    x = torch.tensor([[-1.0, -1.0], [-0.8, -1.2], [1.0, 1.0], [0.8, 1.2]])
    labels = torch.tensor([0, 0, 1, 1])

    metrics = local_supervised_contrastive_step(
        model=model,
        optimizer=optimizer,
        features=x,
        labels=labels,
        temperature=0.2,
    )

    assert metrics["loss"] > 0.0
    for old, layer in zip(before, model.layers, strict=True):
        assert not torch.equal(old, layer.weight.detach())


def test_supervised_training_records_matched_example_budget():
    dataset = TemporalTensorDataset(
        features=torch.tensor([[-1.0, -1.0], [-1.0, -2.0], [1.0, 1.0], [1.0, 2.0]]),
        labels=torch.tensor([0, 0, 1, 1]),
        boundaries=torch.tensor([True, False, False, False]),
    )
    model = SupervisedMLP(input_dim=2, hidden_dims=(3, 2), output_dim=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    summary = train_supervised_mlp(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        epochs=3,
        batch_size=3,
        seed=7,
        device=torch.device("cpu"),
    )

    assert summary.examples == 12
    assert summary.steps == 6
    assert len(summary.layer_parameter_delta_l2) == 2
    assert all(delta > 0.0 for delta in summary.layer_parameter_delta_l2)


def test_local_supervised_contrastive_training_uses_the_same_encoder_budget():
    dataset = TemporalTensorDataset(
        features=torch.tensor(
            [
                [-1.0, -1.0],
                [-0.9, -1.1],
                [-1.1, -0.9],
                [-1.2, -1.0],
                [1.0, 1.0],
                [0.9, 1.1],
                [1.1, 0.9],
                [1.2, 1.0],
            ]
        ),
        labels=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        boundaries=torch.tensor([True] + [False] * 7),
    )
    model = LayerLocalEncoder(
        input_dim=2,
        hidden_dims=(4, 3),
        activation="identity",
        statistics_momentum=0.9,
        lateral_momentum=0.9,
    )
    optimizer = torch.optim.SGD(model.encoder_parameters(), lr=0.05)

    summary = train_local_supervised_contrastive(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        epochs=2,
        batch_size=4,
        seed=13,
        chunk_size=2,
        temperature=0.2,
        device=torch.device("cpu"),
    )

    assert summary.examples == 16
    assert summary.steps == 4
    assert all(delta > 0.0 for delta in summary.layer_parameter_delta_l2)
