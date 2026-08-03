import numpy as np
import torch

from terel.resubmission.data import TemporalTensorDataset
from terel.resubmission.evaluation import (
    class_structure_diagnostics,
    classification_metrics,
    extract_representations,
    fit_linear_probe,
    representation_diagnostics,
)
from terel.resubmission.model import LayerLocalEncoder


def test_representation_extraction_supports_matched_last_and_all_layer_readouts():
    model = LayerLocalEncoder(
        input_dim=2,
        hidden_dims=(2, 2),
        activation="identity",
        statistics_momentum=0.9,
        lateral_momentum=0.9,
    )
    with torch.no_grad():
        for layer in model.layers:
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()
    dataset = TemporalTensorDataset(
        features=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        labels=torch.tensor([0, 1, 0]),
        boundaries=torch.tensor([True, False, False]),
    )

    last = extract_representations(model, dataset, batch_size=2, device=torch.device("cpu"))
    all_layers = extract_representations(
        model,
        dataset,
        batch_size=2,
        device=torch.device("cpu"),
        use_all_layers=True,
    )

    assert torch.equal(last, dataset.features)
    assert torch.equal(all_layers, torch.cat([dataset.features, dataset.features], dim=1))
    assert model.training is False


def test_fixed_linear_probe_and_metrics_recover_a_separable_problem():
    features = torch.tensor(
        [[-2.0, -1.0], [-1.0, -2.0], [-2.0, -2.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]]
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])

    probe, summary = fit_linear_probe(
        features,
        labels,
        num_classes=2,
        seed=17,
        epochs=80,
        batch_size=3,
        optimizer_name="sgd",
        learning_rate=0.1,
        weight_decay=0.0,
        device=torch.device("cpu"),
    )
    metrics = classification_metrics(probe(features), labels, num_classes=2)

    assert summary.examples == 80 * len(features)
    assert summary.steps == 160
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert np.asarray(metrics["confusion_matrix"]).tolist() == [[3, 0], [0, 3]]


def test_representation_diagnostics_respect_stream_boundaries():
    representations = torch.tensor([[0.0, 0.0], [1.0, 2.0], [10.0, 20.0], [11.0, 22.0]])
    boundaries = torch.tensor([True, False, True, False])

    diagnostics = representation_diagnostics(representations, boundaries)

    assert np.isclose(diagnostics["temporal_slowness"], 2.5)
    assert diagnostics["median_feature_variance"] > 0.0
    assert np.isclose(diagnostics["mean_absolute_offdiagonal_correlation"], 1.0)
    assert np.isclose(diagnostics["effective_rank"], 1.0)
    assert diagnostics["active_feature_fraction"] == 1.0


def test_class_structure_diagnostics_quantify_separation_and_selectivity():
    representations = torch.tensor(
        [[-2.0, -1.0], [-1.0, -1.0], [-2.0, -2.0], [2.0, 1.0], [1.0, 1.0], [2.0, 2.0]]
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])

    diagnostics = class_structure_diagnostics(representations, labels, num_classes=2)

    assert diagnostics["between_within_scatter_ratio"] > 5.0
    assert diagnostics["nearest_centroid_accuracy"] == 1.0
    assert diagnostics["median_unit_class_selectivity"] > 1.0
    assert diagnostics["mean_prototype_cosine"] < 0.0
