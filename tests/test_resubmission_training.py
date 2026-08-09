import torch
import pytest

from terel.resubmission import training
from terel.resubmission.data import TemporalTensorDataset
from terel.resubmission.model import LayerLocalEncoder
from terel.resubmission.objective import LossCoefficients
from terel.resubmission.training import (
    augment_mnist_batch,
    local_train_step,
    train_local_encoder,
)


def test_leaky_relu_keeps_a_gradient_path_for_negative_units():
    model = LayerLocalEncoder(
        input_dim=1,
        hidden_dims=(1,),
        activation="leaky_relu",
        statistics_momentum=0.9,
        lateral_momentum=0.99,
    )
    with torch.no_grad():
        model.layers[0].weight.fill_(1.0)
        model.layers[0].bias.zero_()

    output = model(torch.tensor([[-2.0]], requires_grad=True))
    output.sum().backward()

    assert torch.isclose(output[0, 0], torch.tensor(-0.02))
    assert torch.isclose(model.layers[0].weight.grad[0, 0], torch.tensor(-0.02))


def test_one_local_step_changes_every_declared_encoder_layer():
    """Returning early from optimizer registration or loss construction must fail."""
    torch.manual_seed(7)
    model = LayerLocalEncoder(
        input_dim=2,
        hidden_dims=(3, 2),
        activation="identity",
        statistics_momentum=0.9,
        lateral_momentum=0.9,
    )
    optimizer = torch.optim.SGD(model.encoder_parameters(), lr=0.05)
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    boundaries = torch.tensor([True, False, False, False])
    before = [layer.weight.detach().clone() for layer in model.layers]

    local_train_step(
        model=model,
        optimizer=optimizer,
        x=x,
        boundaries=boundaries,
        coefficients=LossCoefficients(similarity=1.0, variance=0.0, covariance=0.0),
        variance_target=1.0,
        detach_previous=True,
    )

    for old, layer in zip(before, model.layers, strict=True):
        assert not torch.equal(old, layer.weight.detach())


def test_later_layer_loss_has_no_gradient_path_to_earlier_layer():
    """Removing the activation detachment must violate spatial locality."""
    model = LayerLocalEncoder(
        input_dim=2,
        hidden_dims=(3, 2),
        activation="identity",
        statistics_momentum=0.9,
        lateral_momentum=0.9,
    )
    activations = model.forward_local(torch.tensor([[1.0, -1.0]]))

    earlier_gradient = torch.autograd.grad(
        activations[1].sum(),
        model.layers[0].weight,
        allow_unused=True,
    )[0]

    assert earlier_gradient is None


def test_direct_covariance_control_updates_correlated_encoder():
    """The direct-covariance ablation must affect encoder weights without a lateral proxy."""
    model = LayerLocalEncoder(
        input_dim=2,
        hidden_dims=(2,),
        activation="identity",
        statistics_momentum=0.9,
        lateral_momentum=0.9,
    )
    with torch.no_grad():
        model.layers[0].weight.copy_(torch.eye(2))
        model.layers[0].bias.zero_()
    optimizer = torch.optim.SGD(model.encoder_parameters(), lr=0.05)
    before = model.layers[0].weight.detach().clone()

    local_train_step(
        model=model,
        optimizer=optimizer,
        x=torch.tensor([[1.0, 1.0], [-1.0, -1.0]]),
        boundaries=torch.tensor([True, False]),
        coefficients=LossCoefficients(similarity=0.0, variance=0.0, covariance=1.0),
        variance_target=1.0,
        detach_previous=True,
        covariance_mode="direct",
    )

    assert not torch.equal(before, model.layers[0].weight.detach())


def test_lateral_proxy_diagnostics_identify_an_exact_direct_direction():
    """Using the current covariance as the proxy must report exact alignment."""
    assert hasattr(training, "lateral_proxy_diagnostics")
    activations = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    mean = torch.zeros(2)
    exact_offdiagonal_covariance = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    diagnostics = training.lateral_proxy_diagnostics(
        activations,
        mean=mean,
        lateral=exact_offdiagonal_covariance,
    )

    assert diagnostics["valid"] is True
    assert diagnostics["cosine_alignment"] == pytest.approx(1.0)
    assert diagnostics["relative_error"] == pytest.approx(0.0)
    assert diagnostics["norm_ratio"] == pytest.approx(1.0)


def test_shifted_proxy_uses_stored_previous_centered_activation():
    """The asynchronous path must be executable from stored state alone."""
    model = LayerLocalEncoder(
        input_dim=2,
        hidden_dims=(2,),
        activation="identity",
        statistics_momentum=0.9,
        lateral_momentum=0.9,
    )
    with torch.no_grad():
        model.layers[0].weight.copy_(torch.eye(2))
        model.layers[0].bias.zero_()
        model.states[0].lateral.copy_(torch.eye(2))
        model.states[0].ensure_previous_centered().copy_(torch.tensor([3.0, 4.0]))
        model.states[0].has_previous.fill_(True)
    optimizer = torch.optim.SGD(model.encoder_parameters(), lr=0.05)
    before = model.layers[0].weight.detach().clone()

    local_train_step(
        model=model,
        optimizer=optimizer,
        x=torch.tensor([[1.0, 2.0]]),
        boundaries=torch.tensor([False]),
        coefficients=LossCoefficients(similarity=0.0, variance=0.0, covariance=1.0),
        variance_target=1.0,
        detach_previous=True,
        covariance_mode="shifted_proxy",
    )

    assert not torch.equal(before, model.layers[0].weight.detach())


def test_training_loop_records_every_layer_update_and_exact_example_budget():
    """The experiment manifest needs auditable depth fidelity and compute counts."""
    torch.manual_seed(29)
    dataset = TemporalTensorDataset(
        features=torch.randn(12, 4),
        labels=torch.arange(12) % 2,
        boundaries=torch.tensor([True] + [False] * 11),
    )
    model = LayerLocalEncoder(
        input_dim=4,
        hidden_dims=(5, 3),
        activation="identity",
        statistics_momentum=0.9,
        lateral_momentum=0.9,
    )
    optimizer = torch.optim.AdamW(model.encoder_parameters(), lr=1e-3)

    summary = train_local_encoder(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        epochs=2,
        batch_size=5,
        order_mode="chronological",
        order_seed=101,
        chunk_size=2,
        coefficients=LossCoefficients(),
        variance_target=1.0,
        detach_previous=True,
        covariance_mode="proxy",
        device=torch.device("cpu"),
    )

    assert summary.examples == 24
    assert summary.steps == 6
    assert summary.epochs == 2
    assert len(summary.layer_parameter_delta_l2) == 2
    assert all(delta > 0.0 for delta in summary.layer_parameter_delta_l2)
    assert len(summary.layer_lateral_delta_l2) == 2
    assert all(delta > 0.0 for delta in summary.layer_lateral_delta_l2)
    assert summary.dynamic_state_numel == sum(state.dynamic_state_numel() for state in model.states)
    assert summary.causal_dynamic_state_numel == sum(
        state.causal_dynamic_state_numel() for state in model.states
    )
    assert summary.auxiliary_parameter_numel == sum(
        state.auxiliary_parameter_numel() for state in model.states
    )
    assert summary.seconds > 0.0


def test_training_loop_records_lagged_proxy_alignment_only_when_requested():
    """The review audit must summarize real minibatch proxy directions without changing training."""
    torch.manual_seed(31)
    dataset = TemporalTensorDataset(
        features=torch.randn(12, 4),
        labels=torch.arange(12) % 2,
        boundaries=torch.tensor([True] + [False] * 11),
    )
    model = LayerLocalEncoder(
        input_dim=4,
        hidden_dims=(5, 3),
        activation="identity",
        statistics_momentum=0.9,
        lateral_momentum=0.0,
    )
    optimizer = torch.optim.AdamW(model.encoder_parameters(), lr=1e-3)

    summary = train_local_encoder(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        epochs=1,
        batch_size=4,
        order_mode="chronological",
        order_seed=101,
        chunk_size=2,
        coefficients=LossCoefficients(),
        variance_target=1.0,
        detach_previous=False,
        covariance_mode="proxy",
        device=torch.device("cpu"),
        audit_lateral_proxy=True,
    )

    assert summary.lateral_proxy_audited_batches == (2, 2)
    assert all(-1.0 <= value <= 1.0 for value in summary.lateral_proxy_cosine_mean)
    assert all(value >= 0.0 for value in summary.lateral_proxy_relative_error_mean)
    assert all(value >= 0.0 for value in summary.lateral_proxy_norm_ratio_mean)


def test_batch_normalization_parameters_are_registered_and_trainable():
    torch.manual_seed(37)
    model = LayerLocalEncoder(
        input_dim=4,
        hidden_dims=(5, 3),
        activation="relu",
        normalization="batch_norm",
        statistics_momentum=0.9,
        lateral_momentum=0.9,
    )
    optimizer = torch.optim.SGD(model.encoder_parameters(), lr=0.02)
    before = [normalization.weight.detach().clone() for normalization in model.normalizations]

    local_train_step(
        model=model,
        optimizer=optimizer,
        x=torch.randn(8, 4),
        boundaries=torch.tensor([True, False, False, False, True, False, False, False]),
        coefficients=LossCoefficients(similarity=1.0, variance=1.0, covariance=0.0),
        variance_target=1.0,
        detach_previous=False,
    )

    assert all(
        not torch.equal(old, normalization.weight.detach())
        for old, normalization in zip(before, model.normalizations, strict=True)
    )


def test_mnist_augmentation_is_seeded_train_only_and_shape_preserving():
    features = torch.linspace(-0.4, 2.0, steps=2 * 28 * 28).reshape(2, 28 * 28)

    first = augment_mnist_batch(features, seed=41)
    second = augment_mnist_batch(features, seed=41)
    different = augment_mnist_batch(features, seed=42)

    assert first.shape == features.shape
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert not torch.equal(first, features)


def test_streaming_normalization_updates_bounded_detached_state_at_batch_one():
    model = LayerLocalEncoder(
        input_dim=3,
        hidden_dims=(4,),
        activation="relu",
        normalization="streaming_norm",
        normalization_momentum=0.9,
        statistics_momentum=0.9,
        lateral_momentum=0.9,
    )
    normalization = model.normalizations[0]
    before = normalization.running_mean.clone()

    output = model.forward_local(torch.tensor([[1.0, -1.0, 0.5]]))[0]

    assert output.shape == (1, 4)
    assert torch.isfinite(output).all()
    assert not torch.equal(before, normalization.running_mean)
    assert normalization.running_mean.grad_fn is None
    assert normalization.running_variance.grad_fn is None


def test_gradient_accumulation_releases_each_graph_and_counts_optimizer_steps():
    torch.manual_seed(43)
    dataset = TemporalTensorDataset(
        features=torch.randn(12, 4),
        labels=torch.arange(12) % 2,
        boundaries=torch.tensor([True] + [False] * 11),
    )
    model = LayerLocalEncoder(
        input_dim=4,
        hidden_dims=(5, 3),
        activation="leaky_relu",
        statistics_momentum=0.9,
        lateral_momentum=0.9,
    )
    optimizer = torch.optim.AdamW(model.encoder_parameters(), lr=1e-3)

    summary = train_local_encoder(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        epochs=2,
        batch_size=1,
        order_mode="chronological",
        order_seed=101,
        chunk_size=2,
        coefficients=LossCoefficients(),
        variance_target=1.0,
        detach_previous=True,
        covariance_mode="proxy",
        device=torch.device("cpu"),
        gradient_accumulation_steps=4,
    )

    assert summary.steps == 24
    assert summary.optimizer_steps == 6
    assert summary.gradient_accumulation_steps == 4
    assert all(delta > 0.0 for delta in summary.layer_parameter_delta_l2)


def test_dual_lateral_rule_preserves_representation_state_and_learns_from_residuals():
    """Conflating representation and residual moments must break the two-state rule."""
    model = LayerLocalEncoder(
        input_dim=2,
        hidden_dims=(2,),
        activation="identity",
        normalization="none",
        statistics_momentum=0.9,
        lateral_momentum=0.0,
    )
    with torch.no_grad():
        model.layers[0].weight.copy_(torch.eye(2))
        model.layers[0].bias.zero_()
        model.states[0].previous.zero_()
        model.states[0].has_previous.fill_(True)
        model.states[0].lateral.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    initializer = getattr(model.states[0], "ensure_residual_lateral", None)
    assert callable(initializer), "separate residual lateral state is not implemented"
    residual_lateral = initializer()
    with torch.no_grad():
        residual_lateral.copy_(torch.ones(2, 2))
    optimizer = torch.optim.SGD(model.encoder_parameters(), lr=0.1)

    local_train_step(
        model=model,
        optimizer=optimizer,
        x=torch.tensor([[1.0, 2.0]]),
        boundaries=torch.tensor([False]),
        coefficients=LossCoefficients(similarity=1.0, variance=0.0, covariance=2.0),
        variance_target=1.0,
        detach_previous=True,
        covariance_mode="residual_state",
        residual_lateral_rule="dual_inhibitory",
        residual_lateral_coefficient=1.0,
        residual_lateral_steps=80,
        residual_lateral_step_size=0.2,
        residual_lateral_include_diagonal=True,
    )

    assert torch.allclose(
        model.layers[0].weight,
        torch.tensor([[0.9, -0.2], [-0.1, 0.8]]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        model.states[0].lateral,
        torch.tensor([[0.0, 2.0], [2.0, 0.0]]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        model.states[0].residual_lateral,
        torch.ones(2, 2),
        atol=1e-6,
        rtol=1e-6,
    )


def test_zero_coupling_residual_state_matches_fixed_affine_streaming_norm_update():
    """The residual target must retain the exact TeReL-S weight gradient."""
    torch.manual_seed(53)
    reference = LayerLocalEncoder(
        input_dim=2,
        hidden_dims=(2,),
        activation="leaky_relu",
        normalization="streaming_norm",
        normalization_momentum=0.9,
        normalization_affine=False,
        statistics_momentum=0.9,
        lateral_momentum=0.9,
    )
    residual = LayerLocalEncoder(
        input_dim=2,
        hidden_dims=(2,),
        activation="leaky_relu",
        normalization="streaming_norm",
        normalization_momentum=0.9,
        normalization_affine=False,
        statistics_momentum=0.9,
        lateral_momentum=0.9,
    )
    residual.load_state_dict(reference.state_dict())
    with torch.no_grad():
        for model in (reference, residual):
            model.states[0].mean.copy_(torch.tensor([0.2, -0.3]))
            model.states[0].variance.copy_(torch.tensor([0.4, 0.8]))
            model.states[0].lateral.copy_(torch.tensor([[0.0, 0.3], [0.3, 0.0]]))
            model.states[0].previous.copy_(torch.tensor([0.1, -0.2]))
            model.states[0].has_previous.fill_(True)

    x = torch.tensor([[0.4, -0.7]])
    boundaries = torch.tensor([False])
    coefficients = LossCoefficients(similarity=1.0, variance=2.5, covariance=1.0)
    reference_optimizer = torch.optim.SGD(reference.encoder_parameters(), lr=0.03)
    residual_optimizer = torch.optim.SGD(residual.encoder_parameters(), lr=0.03)

    local_train_step(
        model=reference,
        optimizer=reference_optimizer,
        x=x,
        boundaries=boundaries,
        coefficients=coefficients,
        variance_target=1.0,
        detach_previous=True,
        covariance_mode="proxy",
    )
    local_train_step(
        model=residual,
        optimizer=residual_optimizer,
        x=x,
        boundaries=boundaries,
        coefficients=coefficients,
        variance_target=1.0,
        detach_previous=True,
        covariance_mode="residual_state",
        residual_lateral_rule="dual_inhibitory",
        residual_lateral_coefficient=0.0,
        residual_lateral_steps=4,
    )

    assert tuple(reference.normalizations[0].parameters()) == ()
    assert tuple(residual.normalizations[0].parameters()) == ()
    assert torch.allclose(
        residual.layers[0].weight,
        reference.layers[0].weight,
        atol=1e-7,
        rtol=1e-6,
    )
    assert torch.allclose(
        residual.layers[0].bias,
        reference.layers[0].bias,
        atol=1e-7,
        rtol=1e-6,
    )
