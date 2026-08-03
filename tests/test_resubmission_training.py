import torch

from terel.resubmission.data import TemporalTensorDataset
from terel.resubmission.model import LayerLocalEncoder
from terel.resubmission.objective import LossCoefficients
from terel.resubmission.training import augment_mnist_batch, local_train_step, train_local_encoder


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
        model.states[0].previous_centered.copy_(torch.tensor([3.0, 4.0]))
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
    assert summary.seconds > 0.0


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
