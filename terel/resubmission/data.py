from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import urllib.request
import zipfile

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


PAMAP2_ACTIVITY_IDS = (1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24)


@dataclass(frozen=True)
class DatasetSplits:
    train: "TemporalTensorDataset"
    validation: "TemporalTensorDataset"
    test: "TemporalTensorDataset"
    metadata: dict


class _EncoderView(Dataset):
    def __init__(self, source):
        self.source = source

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        return self.source.features[index], self.source.boundaries[index]


class TemporalTensorDataset(Dataset):
    """Temporal samples with a label-free view for encoder pretraining."""

    def __init__(self, *, features: torch.Tensor, labels: torch.Tensor, boundaries: torch.Tensor):
        if len(features) != len(labels) or len(features) != len(boundaries):
            raise ValueError("features, labels, and boundaries must have equal length")
        if boundaries.dtype is not torch.bool:
            raise TypeError("boundaries must be boolean")
        self.features = features
        self.labels = labels
        self.boundaries = boundaries

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.boundaries[index]

    def encoder_view(self) -> Dataset:
        return _EncoderView(self)


def _single_stream_boundaries(length: int) -> torch.Tensor:
    boundaries = torch.zeros(length, dtype=torch.bool)
    if length:
        boundaries[0] = True
    return boundaries


def mnist_protocol_from_tensors(
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    validation_size: int,
    seed: int,
) -> DatasetSplits:
    """Create the fixed train/validation/test protocol from official MNIST tensors."""
    train_indices, validation_indices = stratified_split_indices(
        train_labels.detach().cpu().numpy(),
        validation_size=validation_size,
        seed=seed,
    )

    def features(images):
        flattened = images.to(torch.float32).reshape(len(images), -1) / 255.0
        return (flattened - 0.1307) / 0.3081

    official_train_features = features(train_images)
    official_test_features = features(test_images)

    def subset(indices):
        index = torch.as_tensor(indices, dtype=torch.long)
        return TemporalTensorDataset(
            features=official_train_features[index],
            labels=train_labels[index].to(torch.long),
            boundaries=_single_stream_boundaries(len(index)),
        )

    test = TemporalTensorDataset(
        features=official_test_features,
        labels=test_labels.to(torch.long),
        boundaries=_single_stream_boundaries(len(test_labels)),
    )
    return DatasetSplits(
        train=subset(train_indices),
        validation=subset(validation_indices),
        test=test,
        metadata={
            "dataset": "MNIST",
            "split_seed": int(seed),
            "validation_source": "official_train",
            "source_train_rows": int(len(train_labels)),
            "source_test_rows": int(len(test_labels)),
        },
    )


def split_pamap2_subjects(subject_ids):
    subject_ids = tuple(sorted(int(subject_id) for subject_id in subject_ids))
    required = tuple(range(1, 10))
    if subject_ids != required:
        raise ValueError(f"PAMAP2 protocol requires subjects {required}, got {subject_ids}")
    return tuple(range(1, 7)), (7,), (8,)


def prepare_pamap2_subject(raw: np.ndarray, *, stride: int):
    """Filter/downsample one subject while retaining discontinuity boundaries."""
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError("PAMAP2 rows must contain timestamp, activity, and sensor columns")
    if stride <= 0:
        raise ValueError("stride must be positive")

    activity = raw[:, 1].astype(np.int64, copy=False)
    valid_indices = np.flatnonzero(np.isin(activity, PAMAP2_ACTIVITY_IDS))
    selected = valid_indices[::stride]
    if len(selected) == 0:
        return (
            np.empty((0, raw.shape[1] - 2), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=bool),
        )

    features = raw[selected, 2:].astype(np.float32, copy=False)
    label_lookup = {activity_id: index for index, activity_id in enumerate(PAMAP2_ACTIVITY_IDS)}
    labels = np.asarray([label_lookup[int(value)] for value in activity[selected]], dtype=np.int64)
    boundaries = np.zeros(len(selected), dtype=bool)
    boundaries[0] = True
    boundaries[1:] = np.diff(selected) > stride

    timestamps = raw[:, 0]
    finite_deltas = np.diff(timestamps[np.isfinite(timestamps)])
    finite_deltas = finite_deltas[finite_deltas > 0]
    if len(finite_deltas):
        nominal_step = float(np.median(finite_deltas))
        time_gaps = np.diff(timestamps[selected]) > (1.5 * stride * nominal_step)
        boundaries[1:] |= time_gaps
    return features, labels, boundaries


def fit_standardizer(features: np.ndarray):
    features = np.asarray(features, dtype=np.float32)
    mean = np.nanmean(features, axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    scale = np.nanstd(features, axis=0).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale >= 1e-6), scale, 1.0).astype(np.float32)
    return mean, scale


def apply_standardizer(features: np.ndarray, mean: np.ndarray, scale: np.ndarray):
    features = np.asarray(features, dtype=np.float32).copy()
    missing = ~np.isfinite(features)
    if missing.any():
        features[missing] = np.broadcast_to(mean, features.shape)[missing]
    return (features - mean) / scale


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pamap2_protocol_root(root: Path, *, allow_download: bool):
    candidates = (root / "PAMAP2_Dataset" / "Protocol", root / "Protocol")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    if not allow_download:
        raise FileNotFoundError(f"PAMAP2 Protocol directory not found under {root}")

    root.mkdir(parents=True, exist_ok=True)
    archive = root / "PAMAP2_Dataset.zip"
    if not archive.exists():
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip",
            archive,
        )
    with zipfile.ZipFile(archive) as bundle:
        destination = root.resolve()
        for member in bundle.infolist():
            target = (root / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"Unsafe path in PAMAP2 archive: {member.filename}")
        bundle.extractall(root)
    if not candidates[0].is_dir():
        raise FileNotFoundError("Downloaded PAMAP2 archive did not contain PAMAP2_Dataset/Protocol")
    return candidates[0]


def _pamap2_subject_id(path: Path):
    match = re.fullmatch(r"subject(\d+)", path.stem.lower())
    if match is None:
        raise ValueError(f"Cannot parse PAMAP2 subject from {path.name}")
    encoded = int(match.group(1))
    return encoded - 100 if 101 <= encoded <= 109 else encoded


def _concatenate_prepared_subjects(prepared, subject_ids):
    features = []
    labels = []
    boundaries = []
    for subject_id in subject_ids:
        subject_features, subject_labels, subject_boundaries = prepared[subject_id]
        subject_boundaries = subject_boundaries.copy()
        if len(subject_boundaries):
            subject_boundaries[0] = True
        features.append(subject_features)
        labels.append(subject_labels)
        boundaries.append(subject_boundaries)
    return (
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(boundaries, axis=0),
    )


def load_pamap2_protocol(root, *, stride: int = 10, allow_download: bool = True) -> DatasetSplits:
    """Load the frozen subject-disjoint PAMAP2 protocol without label leakage."""
    protocol_root = _pamap2_protocol_root(Path(root), allow_download=allow_download)
    files = sorted(protocol_root.glob("subject*.dat"))
    by_subject = {_pamap2_subject_id(path): path for path in files}
    train_subjects, validation_subjects, test_subjects = split_pamap2_subjects(by_subject)

    prepared = {}
    for subject_id, path in by_subject.items():
        raw = np.loadtxt(path, dtype=np.float32)
        prepared[subject_id] = prepare_pamap2_subject(raw, stride=stride)

    train_arrays = _concatenate_prepared_subjects(prepared, train_subjects)
    validation_arrays = _concatenate_prepared_subjects(prepared, validation_subjects)
    test_arrays = _concatenate_prepared_subjects(prepared, test_subjects)
    mean, scale = fit_standardizer(train_arrays[0])

    def dataset(arrays):
        features, labels, boundaries = arrays
        return TemporalTensorDataset(
            features=torch.from_numpy(apply_standardizer(features, mean, scale)).to(torch.float32),
            labels=torch.from_numpy(labels).to(torch.long),
            boundaries=torch.from_numpy(boundaries).to(torch.bool),
        )

    return DatasetSplits(
        train=dataset(train_arrays),
        validation=dataset(validation_arrays),
        test=dataset(test_arrays),
        metadata={
            "dataset": "PAMAP2",
            "stride": int(stride),
            "train_subjects": list(train_subjects),
            "validation_subjects": list(validation_subjects),
            "test_subjects": list(test_subjects),
            "excluded_subjects": [9],
            "source_sha256": {path.name: _sha256(path) for path in files},
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
        },
    )


def load_mnist_protocol(
    root,
    *,
    validation_size: int = 10_000,
    seed: int = 1701,
    allow_download: bool = True,
) -> DatasetSplits:
    """Load MNIST with validation drawn strictly from the official training split."""
    from torchvision.datasets import MNIST

    train = MNIST(root=str(root), train=True, download=allow_download)
    test = MNIST(root=str(root), train=False, download=allow_download)
    splits = mnist_protocol_from_tensors(
        train.data,
        train.targets,
        test.data,
        test.targets,
        validation_size=validation_size,
        seed=seed,
    )
    raw_files = sorted(Path(train.raw_folder).glob("*.gz"))
    return DatasetSplits(
        train=splits.train,
        validation=splits.validation,
        test=splits.test,
        metadata={
            **splits.metadata,
            "source_sha256": {path.name: _sha256(path) for path in raw_files},
        },
    )


def concatenate_subject_streams(streams, *, subject_ids):
    feature_parts = []
    label_parts = []
    boundary_parts = []
    for subject_id in subject_ids:
        if subject_id not in streams:
            raise KeyError(f"Missing subject {subject_id}")
        features, labels = streams[subject_id]
        features = np.asarray(features)
        labels = np.asarray(labels)
        if len(features) != len(labels) or len(features) == 0:
            raise ValueError(f"Invalid stream for subject {subject_id}")
        boundaries = np.zeros(len(features), dtype=bool)
        boundaries[0] = True
        feature_parts.append(features)
        label_parts.append(labels)
        boundary_parts.append(boundaries)
    return (
        np.concatenate(feature_parts, axis=0),
        np.concatenate(label_parts, axis=0),
        np.concatenate(boundary_parts, axis=0),
    )


def class_chunk_order(labels, *, chunk_size: int, seed: int):
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    rng = np.random.default_rng(seed)
    chunks = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        chunks.extend(indices[start : start + chunk_size] for start in range(0, len(indices), chunk_size))
    rng.shuffle(chunks)
    order = np.concatenate(chunks)
    boundaries = np.zeros(len(order), dtype=bool)
    offset = 0
    for chunk in chunks:
        boundaries[offset] = True
        offset += len(chunk)
    return order, boundaries


def encoder_batches(
    dataset: TemporalTensorDataset,
    *,
    batch_size: int,
    order_mode: str,
    seed: int,
    chunk_size: int,
):
    """Yield label-free batches while making every invalid temporal edge explicit."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if order_mode == "chronological":
        order = np.arange(len(dataset))
        boundaries = dataset.boundaries
    elif order_mode == "shuffled":
        order = np.random.default_rng(seed).permutation(len(dataset))
        boundaries = torch.zeros(len(dataset), dtype=torch.bool)
        if len(dataset):
            boundaries[0] = True
    elif order_mode == "class_chunks":
        order, boundary_array = class_chunk_order(
            dataset.labels.detach().cpu().numpy(),
            chunk_size=chunk_size,
            seed=seed,
        )
        boundaries = torch.from_numpy(boundary_array)
    else:
        raise ValueError(f"Unknown order mode: {order_mode}")

    index = torch.as_tensor(order, dtype=torch.long)
    ordered_features = dataset.features[index]
    if order_mode == "chronological":
        ordered_boundaries = boundaries[index]
    else:
        ordered_boundaries = boundaries
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        yield ordered_features[start:stop], ordered_boundaries[start:stop]


def stratified_split_indices(labels, *, validation_size: int, seed: int):
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not 0 < validation_size < labels.size:
        raise ValueError("validation_size must lie between zero and the dataset size")
    indices = np.arange(labels.size)
    train, validation = train_test_split(
        indices,
        test_size=validation_size,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return np.sort(train), np.sort(validation)
