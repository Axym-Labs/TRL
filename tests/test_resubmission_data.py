import numpy as np
import torch

from terel.resubmission.data import (
    DatasetSplits,
    TemporalTensorDataset,
    apply_standardizer,
    class_chunk_order,
    concatenate_subject_streams,
    encoder_batches,
    fit_standardizer,
    load_pamap2_protocol,
    mnist_protocol_from_tensors,
    prepare_pamap2_subject,
    split_pamap2_subjects,
    stratified_split_indices,
)


def test_stratified_train_validation_split_is_disjoint_and_class_balanced():
    """Tuning rows must not overlap training rows or silently lose a class."""
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    train, validation = stratified_split_indices(labels, validation_size=4, seed=1701)

    assert len(train) == 4
    assert len(validation) == 4
    assert set(train).isdisjoint(set(validation))
    assert set(train) | set(validation) == set(range(8))
    assert np.bincount(labels[train], minlength=2).tolist() == [2, 2]
    assert np.bincount(labels[validation], minlength=2).tolist() == [2, 2]


def test_encoder_view_cannot_yield_downstream_labels():
    """Natural-stream pretraining must have no label field to read accidentally."""
    dataset = TemporalTensorDataset(
        features=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        labels=torch.tensor([7, 9]),
        boundaries=torch.tensor([True, False]),
    )

    encoder_item = dataset.encoder_view()[1]
    probe_item = dataset[1]

    assert len(encoder_item) == 2
    assert torch.equal(encoder_item[0], torch.tensor([3.0, 4.0]))
    assert encoder_item[1].item() is False
    assert len(probe_item) == 3
    assert probe_item[1].item() == 9


def test_subject_concatenation_marks_only_real_stream_boundaries():
    """The temporal objective must not connect the last sample of one person to another."""
    streams = {
        1: (np.array([[1.0], [2.0]]), np.array([0, 0])),
        3: (np.array([[7.0], [8.0], [9.0]]), np.array([1, 1, 2])),
    }

    features, labels, boundaries = concatenate_subject_streams(streams, subject_ids=(1, 3))

    assert features[:, 0].tolist() == [1.0, 2.0, 7.0, 8.0, 9.0]
    assert labels.tolist() == [0, 0, 1, 1, 2]
    assert boundaries.tolist() == [True, False, True, False, False]


def test_class_chunk_order_marks_chunks_and_keeps_labels_coherent():
    """MNIST ordering may use labels, but pair validity must expose each chunk boundary."""
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])

    order, boundaries = class_chunk_order(labels, chunk_size=2, seed=23)

    assert sorted(order.tolist()) == list(range(8))
    assert boundaries.tolist() == [True, False, True, False, True, False, True, False]
    ordered_labels = labels[order]
    for start in range(0, 8, 2):
        assert ordered_labels[start] == ordered_labels[start + 1]


def test_chronological_encoder_batches_preserve_data_boundaries_without_labels():
    """The natural-stream iterator must preserve order and expose only features/boundaries."""
    dataset = TemporalTensorDataset(
        features=torch.arange(10, dtype=torch.float32).reshape(5, 2),
        labels=torch.tensor([4, 4, 8, 8, 8]),
        boundaries=torch.tensor([True, False, True, False, False]),
    )

    batches = list(
        encoder_batches(
            dataset,
            batch_size=2,
            order_mode="chronological",
            seed=5,
            chunk_size=2,
        )
    )

    features = torch.cat([batch[0] for batch in batches])
    boundaries = torch.cat([batch[1] for batch in batches])
    assert torch.equal(features, dataset.features)
    assert torch.equal(boundaries, dataset.boundaries)
    assert all(len(batch) == 2 for batch in batches)


def test_mnist_protocol_uses_only_official_training_rows_for_train_validation():
    """The official test split must never be recycled as validation data."""
    train_images = torch.arange(12 * 4, dtype=torch.uint8).reshape(12, 2, 2)
    train_labels = torch.tensor([0, 1] * 6)
    test_images = torch.arange(4 * 4, dtype=torch.uint8).reshape(4, 2, 2) + 100
    test_labels = torch.tensor([0, 1, 0, 1])

    splits = mnist_protocol_from_tensors(
        train_images,
        train_labels,
        test_images,
        test_labels,
        validation_size=4,
        seed=1701,
    )

    assert isinstance(splits, DatasetSplits)
    assert len(splits.train) == 8
    assert len(splits.validation) == 4
    assert len(splits.test) == 4
    assert splits.test.features.min() > splits.train.features.max()
    assert splits.metadata["source_test_rows"] == 4
    assert splits.metadata["validation_source"] == "official_train"


def test_pamap_subject_preparation_marks_filtered_time_gaps():
    """Removing unlabeled periods must not create false adjacent temporal pairs."""
    raw = np.zeros((8, 5), dtype=np.float64)
    raw[:, 0] = np.arange(8) * 0.01
    raw[:, 1] = np.array([1, 1, 0, 1, 1, 1, 1, 1])
    raw[:, 2:] = np.arange(24).reshape(8, 3)

    features, labels, boundaries = prepare_pamap2_subject(raw, stride=2)

    assert labels.tolist() == [0, 0, 0, 0]
    assert features[:, 0].tolist() == [0.0, 9.0, 15.0, 21.0]
    assert boundaries.tolist() == [True, True, False, False]


def test_pamap_subject_split_and_normalization_are_train_only():
    """Subjects 7/8 remain held out and incomplete subject 9 is excluded a priori."""
    train, validation, test = split_pamap2_subjects(tuple(range(1, 10)))
    assert train == tuple(range(1, 7))
    assert validation == (7,)
    assert test == (8,)

    x_train = np.array([[1.0, np.nan], [3.0, 5.0]], dtype=np.float32)
    mean, scale = fit_standardizer(x_train)
    transformed = apply_standardizer(np.array([[100.0, np.nan]], dtype=np.float32), mean, scale)
    assert np.allclose(mean, [2.0, 5.0])
    assert np.allclose(scale, [1.0, 1.0])
    assert np.allclose(transformed, [[98.0, 0.0]])


def test_pamap_loader_keeps_subject_splits_and_boundaries(tmp_path):
    """The public loader must implement the frozen 1--6/7/8 protocol exactly."""
    protocol_root = tmp_path / "PAMAP2_Dataset" / "Protocol"
    protocol_root.mkdir(parents=True)
    for subject_id in range(1, 10):
        raw = np.zeros((4, 54), dtype=np.float64)
        raw[:, 0] = np.arange(4) * 0.01
        raw[:, 1] = 1
        raw[:, 2:] = subject_id
        np.savetxt(protocol_root / f"subject10{subject_id}.dat", raw)

    splits = load_pamap2_protocol(tmp_path, stride=1, allow_download=False)

    assert len(splits.train) == 24
    assert len(splits.validation) == 4
    assert len(splits.test) == 4
    assert splits.train.boundaries.nonzero().flatten().tolist() == list(range(0, 24, 4))
    assert splits.metadata["train_subjects"] == [1, 2, 3, 4, 5, 6]
    assert splits.metadata["validation_subjects"] == [7]
    assert splits.metadata["test_subjects"] == [8]
    assert splits.metadata["excluded_subjects"] == [9]
    assert len(splits.metadata["source_sha256"]) == 9
