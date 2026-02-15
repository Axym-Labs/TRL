from dataclasses import dataclass
from typing import Callable

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from trl.config.config import DataConfig
from trl.data import CoherentSampler


@dataclass
class DatasetSpec:
    train_dataset_fn: Callable
    val_dataset_fn: Callable
    train_transform_fn: Callable[[bool], transforms.Compose]
    val_transform_fn: Callable[[], transforms.Compose]
    supports_sequence_rows: bool = False


def _mnist_train_transform(augment: bool):
    if augment:
        return transforms.Compose([
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def _mnist_val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def _cifar_train_transform(augment: bool):
    if augment:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


def _cifar_val_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


def seq_transform(x):
    # first dimension because this operates at the dataset level
    # no batch dimension yet
    return x.squeeze(0)


class TimitSequenceDataset(Dataset):
    """
    Wraps torchaudio TIMIT samples into fixed [S, D] sequences.
    Uses speaker id as label if available; otherwise emits label 0.
    """
    def __init__(self, base_ds, frame_len: int = 256, frame_hop: int = 128, max_frames: int = 64):
        self.base_ds = base_ds
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.max_frames = max_frames
        self._speaker_to_idx = {}

    def __len__(self):
        return len(self.base_ds)

    def _speaker_index(self, speaker):
        key = str(speaker)
        if key not in self._speaker_to_idx:
            self._speaker_to_idx[key] = len(self._speaker_to_idx)
        return self._speaker_to_idx[key]

    def _to_sequence(self, wav: torch.Tensor):
        if wav.ndim == 2:
            wav = wav.mean(dim=0)
        elif wav.ndim != 1:
            wav = wav.reshape(-1)
        need = self.frame_len + self.frame_hop * max(0, self.max_frames - 1)
        if wav.numel() < need:
            wav = torch.nn.functional.pad(wav, (0, need - wav.numel()))
        frames = wav.unfold(0, self.frame_len, self.frame_hop)
        frames = frames[: self.max_frames].contiguous()
        if frames.shape[0] < self.max_frames:
            pad = torch.zeros((self.max_frames - frames.shape[0], self.frame_len), dtype=frames.dtype)
            frames = torch.cat([frames, pad], dim=0)
        return frames

    def __getitem__(self, idx):
        sample = self.base_ds[idx]
        # TIMIT tuple is backend/version dependent; waveform is first item.
        wav = sample[0]
        label = 0
        if len(sample) >= 4:
            label = self._speaker_index(sample[3])
        return self._to_sequence(wav), label


def _get_labels(ds):
    if hasattr(ds, "targets"):
        t = ds.targets
        return list(t.tolist()) if hasattr(t, "tolist") else list(t)
    labels = []
    for i in range(len(ds)):
        _x, y = ds[i]
        labels.append(int(y) if not isinstance(y, int) else y)
    return labels


def _make_train_loader(base_train, cfg: DataConfig):
    label_list = _get_labels(base_train)
    sampler = CoherentSampler(label_list=label_list, coherence=cfg.coherence, chunk_size=cfg.chunk_size, shuffle=True)
    return DataLoader(
        base_train,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )


def _make_val_loader(val_ds, cfg: DataConfig):
    return DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )


def _build_torchvision_loaders(cfg: DataConfig, problem_type: str, spec: DatasetSpec, force_rows_sequence: bool = False):
    if force_rows_sequence and problem_type != "sequence":
        raise ValueError("dataset_name='mnist-rows' requires problem_type='sequence'.")
    if problem_type == "sequence" and not (spec.supports_sequence_rows or force_rows_sequence):
        raise ValueError(f"dataset_name='{cfg.dataset_name}' does not support problem_type='sequence'.")

    if_seq_squeeze = lambda tr: transforms.Compose([tr, seq_transform]) if (problem_type == "sequence") else tr
    train_tf = if_seq_squeeze(spec.train_transform_fn(cfg.encoder_augment))
    val_tf = if_seq_squeeze(spec.val_transform_fn())

    base_train = spec.train_dataset_fn(cfg.data_path, train=True, download=True, transform=train_tf)
    head_base_train = spec.train_dataset_fn(cfg.data_path, train=True, download=True, transform=val_tf)
    val_ds = spec.val_dataset_fn(cfg.data_path, train=False, download=True, transform=val_tf)

    train_loader = _make_train_loader(base_train, cfg)
    head_train_loader = _make_train_loader(head_base_train, cfg)
    val_loader = _make_val_loader(val_ds, cfg)
    return train_loader, head_train_loader, val_loader


def _build_timit_loaders(cfg: DataConfig, problem_type: str):
    if problem_type != "sequence":
        raise ValueError("dataset_name='timit' currently supports problem_type='sequence' only.")
    try:
        from torchaudio.datasets import TIMIT
    except ImportError as exc:
        raise ImportError("dataset_name='timit' requires torchaudio to be installed.") from exc

    # TIMIT usually requires a locally available corpus in cfg.data_path.
    train_base = TIMIT(cfg.data_path, subset="train", download=False)
    val_base = TIMIT(cfg.data_path, subset="test", download=False)

    train_ds = TimitSequenceDataset(train_base)
    val_ds = TimitSequenceDataset(val_base)
    train_loader = _make_train_loader(train_ds, cfg)
    head_train_loader = _make_train_loader(train_ds, cfg)
    val_loader = _make_val_loader(val_ds, cfg)
    return train_loader, head_train_loader, val_loader


def build_dataloaders(cfg: DataConfig, problem_type: str):
    name = cfg.dataset_name.lower()
    if name == "mnist":
        spec = DatasetSpec(
            train_dataset_fn=torchvision.datasets.MNIST,
            val_dataset_fn=torchvision.datasets.MNIST,
            train_transform_fn=_mnist_train_transform,
            val_transform_fn=_mnist_val_transform,
            supports_sequence_rows=True,
        )
        return _build_torchvision_loaders(cfg, problem_type, spec)
    if name == "mnist-rows":
        spec = DatasetSpec(
            train_dataset_fn=torchvision.datasets.MNIST,
            val_dataset_fn=torchvision.datasets.MNIST,
            train_transform_fn=_mnist_train_transform,
            val_transform_fn=_mnist_val_transform,
            supports_sequence_rows=True,
        )
        return _build_torchvision_loaders(cfg, problem_type, spec, force_rows_sequence=True)
    if name == "cifar10":
        spec = DatasetSpec(
            train_dataset_fn=torchvision.datasets.CIFAR10,
            val_dataset_fn=torchvision.datasets.CIFAR10,
            train_transform_fn=_cifar_train_transform,
            val_transform_fn=_cifar_val_transform,
            supports_sequence_rows=False,
        )
        return _build_torchvision_loaders(cfg, problem_type, spec)
    if name == "cifar100":
        spec = DatasetSpec(
            train_dataset_fn=torchvision.datasets.CIFAR100,
            val_dataset_fn=torchvision.datasets.CIFAR100,
            train_transform_fn=_cifar_train_transform,
            val_transform_fn=_cifar_val_transform,
            supports_sequence_rows=False,
        )
        return _build_torchvision_loaders(cfg, problem_type, spec)
    if name == "timit":
        return _build_timit_loaders(cfg, problem_type)
    raise ValueError(f"Unknown dataset_name='{cfg.dataset_name}'.")

