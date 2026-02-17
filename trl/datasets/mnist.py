from dataclasses import dataclass
from typing import Callable
from pathlib import Path
from urllib.error import URLError
import wave
import numpy as np

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


class YesNoSequenceDataset(Dataset):
    """
    Wrap YESNO waveforms into fixed [S, D] sequences.
    Label is the 8-bit yes/no pattern collapsed to an integer in [0, 255].
    """
    def __init__(self, base_ds, frame_len: int = 256, frame_hop: int = 128, max_frames: int = 64):
        self.base_ds = base_ds
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.max_frames = max_frames

    def __len__(self):
        return len(self.base_ds)

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

    @staticmethod
    def _labels_to_int(labels):
        # YESNO emits an iterable of 8 binary labels.
        bits = [int(b) for b in labels]
        v = 0
        for b in bits:
            v = (v << 1) | (b & 1)
        return v

    def __getitem__(self, idx):
        wav, _sr, labels = self.base_ds[idx]
        return self._to_sequence(wav), self._labels_to_int(labels)


class SpeechCommandsSequenceDataset(Dataset):
    """
    Wrap SpeechCommands waveforms into fixed [S, D] sequences.
    Label is the command class index.
    """
    def __init__(self, base_ds, frame_len: int = 256, frame_hop: int = 128, max_frames: int = 64):
        self.base_ds = base_ds
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.max_frames = max_frames
        self._label_to_idx = {}

    def __len__(self):
        return len(self.base_ds)

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

    def _label_index(self, label: str):
        key = str(label)
        if key not in self._label_to_idx:
            self._label_to_idx[key] = len(self._label_to_idx)
        return self._label_to_idx[key]

    def __getitem__(self, idx):
        wav, _sr, label, *_ = self.base_ds[idx]
        return self._to_sequence(wav), self._label_index(label)


class LocalYesNoDataset(Dataset):
    """
    Local YESNO loader that parses .wav files directly from disk and avoids
    torchaudio's runtime codec dependencies.
    """
    def __init__(self, root: Path):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.wav"))
        if not self.files:
            raise FileNotFoundError(f"No .wav files found under {self.root}")

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _read_wav(path: Path) -> torch.Tensor:
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        if sample_width != 2:
            raise ValueError(f"Unsupported sample width {sample_width} in {path}; expected 16-bit PCM.")
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        return torch.from_numpy(audio)

    @staticmethod
    def _labels_from_name(path: Path):
        # Filename pattern: "0_1_..._1.wav" (8 binary digits)
        parts = path.stem.split("_")
        if len(parts) != 8 or any(p not in {"0", "1"} for p in parts):
            raise ValueError(f"Invalid YESNO filename pattern: {path.name}")
        return [int(p) for p in parts]

    def __getitem__(self, idx):
        p = self.files[idx]
        wav = self._read_wav(p)
        labels = self._labels_from_name(p)
        # match torchaudio YESNO tuple style: (waveform, sample_rate, labels)
        return wav, 8000, labels


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
    if cfg.use_coherent_sampler:
        label_list = _get_labels(base_train)
        sampler = CoherentSampler(
            label_list=label_list,
            coherence=cfg.coherence,
            chunk_size=cfg.chunk_size,
            shuffle=True,
        )
        return DataLoader(
            base_train,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=True,
            persistent_workers=cfg.num_workers > 0,
        )
    return DataLoader(
        base_train,
        batch_size=cfg.batch_size,
        shuffle=True,
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

    def _instantiate(ds_fn, train: bool, transform):
        if not cfg.allow_download:
            return ds_fn(cfg.data_path, train=train, download=False, transform=transform)
        try:
            return ds_fn(cfg.data_path, train=train, download=True, transform=transform)
        except (URLError, RuntimeError, OSError) as exc:
            # Retry in offline mode for users with manually prepared local datasets.
            try:
                return ds_fn(cfg.data_path, train=train, download=False, transform=transform)
            except Exception:
                name = cfg.dataset_name.lower()
                raise RuntimeError(
                    f"Could not download '{name}' and no local copy was found under '{cfg.data_path}'. "
                    f"Set data_config.allow_download=False if you want strict offline behavior and place the "
                    f"dataset files manually in torchvision's expected folder structure."
                ) from exc

    base_train = _instantiate(spec.train_dataset_fn, train=True, transform=train_tf)
    head_base_train = _instantiate(spec.train_dataset_fn, train=True, transform=val_tf)
    val_ds = _instantiate(spec.val_dataset_fn, train=False, transform=val_tf)

    train_loader = _make_train_loader(base_train, cfg)
    head_train_loader = _make_train_loader(head_base_train, cfg)
    val_loader = _make_val_loader(val_ds, cfg)
    return train_loader, head_train_loader, val_loader


def _build_timit_loaders(cfg: DataConfig, problem_type: str):
    if problem_type != "sequence":
        raise ValueError("dataset_name='timit' currently supports problem_type='sequence' only.")
    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError("dataset_name='timit' requires torchaudio to be installed.") from exc

    # Recent torchaudio releases do not provide a TIMIT downloader/reader class.
    # We therefore support local-corpus loading from:
    #   <data_path>/timit/train/**/*.wav
    #   <data_path>/timit/test/**/*.wav
    root = Path(cfg.data_path) / "timit"
    train_root = root / "train"
    test_root = root / "test"
    if not train_root.exists() or not test_root.exists():
        raise FileNotFoundError(
            "dataset_name='timit' expects a local corpus at "
            f"'{train_root}' and '{test_root}'. "
            "TIMIT is typically licensed and not auto-downloadable."
        )

    class LocalTimitDataset(Dataset):
        def __init__(self, subset_root: Path):
            self.files = sorted(subset_root.rglob("*.wav"))
            if not self.files:
                raise FileNotFoundError(f"No .wav files found under {subset_root}")
            self.speaker_to_idx = {}

        def __len__(self):
            return len(self.files)

        def _speaker_index(self, wav_path: Path):
            # Heuristic: use parent directory as speaker id.
            speaker = wav_path.parent.name
            if speaker not in self.speaker_to_idx:
                self.speaker_to_idx[speaker] = len(self.speaker_to_idx)
            return self.speaker_to_idx[speaker]

        def __getitem__(self, idx):
            wav_path = self.files[idx]
            wav, _sr = torchaudio.load(str(wav_path))
            label = self._speaker_index(wav_path)
            return wav, label

    train_base = LocalTimitDataset(train_root)
    val_base = LocalTimitDataset(test_root)
    train_ds = TimitSequenceDataset(train_base)
    val_ds = TimitSequenceDataset(val_base)
    train_loader = _make_train_loader(train_ds, cfg)
    head_train_loader = _make_train_loader(train_ds, cfg)
    val_loader = _make_val_loader(val_ds, cfg)
    return train_loader, head_train_loader, val_loader


def _build_yesno_loaders(cfg: DataConfig, problem_type: str):
    if problem_type != "sequence":
        raise ValueError("dataset_name='yesno' currently supports problem_type='sequence' only.")
    root = Path(cfg.data_path) / "yesno"
    waves_root = root / "waves_yesno"
    if not waves_root.exists():
        # Optional: try a one-off download path via torchaudio, then proceed local.
        if cfg.allow_download:
            try:
                from torchaudio.datasets import YESNO
                _ = YESNO(str(root), download=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download/load YESNO under '{root}'. "
                    "If you are offline, place YESNO files locally and set data_config.allow_download=False."
                ) from exc
        if not waves_root.exists():
            raise FileNotFoundError(f"Missing YESNO folder: {waves_root}")

    base_ds = LocalYesNoDataset(waves_root)

    # Single corpus split; use deterministic split for quick experiments.
    n = len(base_ds)
    if n < 10:
        raise RuntimeError(f"YESNO dataset too small/unavailable (n={n}).")
    val_n = max(8, int(0.2 * n))
    train_n = n - val_n
    train_base, val_base = torch.utils.data.random_split(
        base_ds,
        lengths=[train_n, val_n],
        generator=torch.Generator().manual_seed(42),
    )
    train_ds = YesNoSequenceDataset(train_base)
    val_ds = YesNoSequenceDataset(val_base)
    train_loader = _make_train_loader(train_ds, cfg)
    head_train_loader = _make_train_loader(train_ds, cfg)
    val_loader = _make_val_loader(val_ds, cfg)
    return train_loader, head_train_loader, val_loader


def _build_speechcommands_loaders(cfg: DataConfig, problem_type: str):
    if problem_type != "sequence":
        raise ValueError("dataset_name='speechcommands' currently supports problem_type='sequence' only.")
    try:
        from torchaudio.datasets import SPEECHCOMMANDS
    except ImportError as exc:
        raise ImportError("dataset_name='speechcommands' requires torchaudio to be installed.") from exc

    root = Path(cfg.data_path) / "speechcommands"
    try:
        train_base = SPEECHCOMMANDS(str(root), subset="training", download=cfg.allow_download)
        val_base = SPEECHCOMMANDS(str(root), subset="validation", download=cfg.allow_download)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load SpeechCommands under '{root}'. "
            "If you are offline, place files locally and set data_config.allow_download=False."
        ) from exc

    train_ds = SpeechCommandsSequenceDataset(train_base)
    val_ds = SpeechCommandsSequenceDataset(val_base)
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
    if name == "yesno":
        return _build_yesno_loaders(cfg, problem_type)
    if name == "speechcommands":
        return _build_speechcommands_loaders(cfg, problem_type)
    raise ValueError(f"Unknown dataset_name='{cfg.dataset_name}'.")

