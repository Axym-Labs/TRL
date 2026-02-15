import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trl.config.config import BatchNormConfig, Config, EncoderConfig
from trl.config.configurations import finish_setup
from trl.datasets.mnist import build_dataloaders
from trl.modules.encoder import TREncoder, TRSeqEncoder, TRSeqElementwiseEncoder
from trl.trainer.encoder import EncoderTrainer, SeqEncoderTrainer
from trl.trainer.head import ClassifierHead, PredictorHead


def load_cfg(hparams_path: Path) -> Config:
    cfg = Config()
    h = torch.load(hparams_path, map_location="cpu", weights_only=False)

    for key, val in h.items():
        if key == "encoders":
            cfg.encoders = [EncoderConfig(**enc) for enc in val]
            continue
        if key == "data_config":
            for k, v in val.items():
                setattr(cfg.data_config, k, v)
            continue
        if key == "trloss_config":
            for k, v in val.items():
                setattr(cfg.trloss_config, k, v)
            continue
        if key == "store_config":
            for k, v in val.items():
                setattr(cfg.store_config, k, v)
            continue
        if key == "batchnorm_config":
            cfg.batchnorm_config = None if val is None else BatchNormConfig(**val)
            continue
        setattr(cfg, key, val)

    finish_setup(cfg)
    return cfg


def infer_encoders_from_state_dict(state_dict: dict) -> list[EncoderConfig]:
    layer_shapes = []
    i = 0
    while True:
        key = f"encoder.layers.{i}.lin.weight"
        if key not in state_dict:
            break
        w = state_dict[key]
        out_dim, in_dim = int(w.shape[0]), int(w.shape[1])
        layer_shapes.append((in_dim, out_dim))
        i += 1
    if not layer_shapes:
        raise ValueError("Could not infer encoder layer dims from checkpoint state_dict.")
    return [EncoderConfig(tuple(layer_shapes))]


def newest_run_dir(base: Path) -> Path:
    run_dirs = [p for p in base.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {base}")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def setup_plot_style():
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.5,
    })


def style_axes(ax, frameless: bool = False):
    if frameless:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    else:
        # Keep only subtle left/bottom spines.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, :2] * s[:2]


def lda_2d(x: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    d = x.shape[1]
    x_mean = x.mean(axis=0, keepdims=True)
    sw = np.zeros((d, d), dtype=np.float64)
    sb = np.zeros((d, d), dtype=np.float64)
    for c in range(num_classes):
        xc = x[y == c]
        if xc.shape[0] < 2:
            continue
        mc = xc.mean(axis=0, keepdims=True)
        centered = xc - mc
        sw += centered.T @ centered
        diff = mc - x_mean
        sb += xc.shape[0] * (diff.T @ diff)

    reg = 1e-4 * np.eye(d)
    m = np.linalg.solve(sw + reg, sb)
    eigvals, eigvecs = np.linalg.eig(m)
    idx = np.argsort(np.real(eigvals))[::-1][:2]
    w = np.real(eigvecs[:, idx])
    return x @ w


def cosine_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def spectral_2d(x: np.ndarray, n_neighbors: int = 12) -> np.ndarray:
    xn = cosine_normalize(x.astype(np.float64))
    sim = xn @ xn.T
    np.fill_diagonal(sim, -np.inf)
    n = sim.shape[0]
    n_neighbors = min(n_neighbors, n - 1)
    knn_idx = np.argpartition(-sim, kth=n_neighbors, axis=1)[:, :n_neighbors]

    w = np.zeros((n, n), dtype=np.float64)
    rows = np.arange(n)[:, None]
    nbr_sim = np.maximum(sim[rows, knn_idx], 0.0)
    w[rows, knn_idx] = nbr_sim
    w = np.maximum(w, w.T)

    d = w.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-8))
    l = np.eye(n) - (d_inv_sqrt[:, None] * w * d_inv_sqrt[None, :])
    eigvals, eigvecs = np.linalg.eigh(l)
    order = np.argsort(eigvals)
    return eigvecs[:, order[1:3]]


def tsne_2d(x: np.ndarray, random_state: int = 42) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    except Exception as exc:
        raise RuntimeError("t-SNE requires scikit-learn. Install with `uv pip install scikit-learn`.") from exc

    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x = x / np.maximum(x_std, 1e-8)

    # Denoise high-dimensional inputs before t-SNE while preserving neighborhood structure.
    pca_dim = min(50, x.shape[1], x.shape[0] - 1)
    if pca_dim >= 2 and x.shape[1] > pca_dim:
        x = PCA(n_components=pca_dim, random_state=random_state).fit_transform(x)

    perplexity = max(15, min(50, x.shape[0] // 200))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        metric="cosine",
        early_exaggeration=18.0,
        random_state=random_state,
    )
    return tsne.fit_transform(x)


def pairwise_cosine_stats(reps: np.ndarray, labels: np.ndarray, sample_pairs: int = 30000) -> dict:
    rng = np.random.default_rng(42)
    n = reps.shape[0]
    idx_a = rng.integers(0, n, size=sample_pairs)
    idx_b = rng.integers(0, n, size=sample_pairs)
    same = labels[idx_a] == labels[idx_b]
    reps_n = cosine_normalize(reps)
    cos = (reps_n[idx_a] * reps_n[idx_b]).sum(axis=1)
    return {
        "same": cos[same],
        "different": cos[~same],
    }


def scatter_latent(ax, z2d, labels, num_classes):
    sc = ax.scatter(z2d[:, 0], z2d[:, 1], c=labels, s=5, cmap="tab10", alpha=0.75, linewidths=0)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(False)
    style_axes(ax, frameless=False)
    return sc


def write_plot_descriptions(out_dir: Path):
    md = """# Figure Descriptions

## `pca_representations.png`
Two-dimensional PCA projection of validation representations. Colors indicate digit classes, showing dominant global variance structure and coarse class alignment.

## `lda_representations.png`
Two-dimensional LDA projection using class labels. This view emphasizes class-discriminative structure and highlights linear separability in the learned latent space.

## `spectral_representations.png`
Laplacian-eigenmap (spectral) projection of latent vectors based on a cosine k-nearest-neighbor graph. This visualization emphasizes local manifold neighborhoods and nonlinear cluster geometry.

## `tsne_representations.png`
Two-dimensional t-SNE projection of latent representations. This view emphasizes neighborhood-preserving nonlinear class structure and local cluster compactness.

## `first_layer_top64_rf.png`
Top-64 first-layer receptive fields selected by weight L2 norm. Robust nonlinear contrast scaling is used to improve visibility of localized and oriented patterns.

## `top_selective_neurons_heatmap.png`
Class-conditional mean activations for the most selective neurons in the representation. Columns are neurons sorted by selectivity; rows are classes.

## `confusion_matrix_normalized.png`
Row-normalized confusion matrix of linear-head predictions on validation data. Diagonal intensity reflects per-class recall.

## `class_prototype_cosine.png`
Cosine similarity matrix of class prototype vectors (class-mean latent embeddings). Off-diagonal values indicate inter-class representational overlap.

## `pairwise_cosine_distribution.png`
Distribution of pairwise cosine similarities for same-class versus different-class sample pairs. The separation margin summarizes latent compactness and class separation.

## `representation_metrics.json`
Scalar summary metrics used for quantitative reporting: head validation accuracy, pairwise cosine margins, prototype overlap, and neuron selectivity statistics.
"""
    (out_dir / "plot_descriptions.md").write_text(md, encoding="utf-8")


def write_sequence_plot_descriptions(out_dir: Path):
    md = """# Figure Descriptions

## `sequence_latent_tsne_samples.png`
t-SNE projection of per-timestep sequence latents. Each point is one timestep representation and colors indicate sequence digit class.

## `sequence_latent_tsne_mean_paths.png`
Smoothed mean trajectories in t-SNE space: one global mean path across all samples and one class-mean path per digit. Markers show the original timestep points that define each path.

## `sequence_latent_tsne_per_digit_paths.png`
Per-digit trajectory panels in t-SNE space. Each panel shows smoothed sample paths for one digit together with the digit mean path and timestep markers.

## `sequence_autoregressive_lower_half.png`
Autoregressive sequence prediction on random digits: the model is conditioned on the upper half of the image (first 14 rows) and then predicts rows 15-28 recursively.

## `sequence_autoregressive_double_rows.png`
Autoregressive generation beyond the original image length: conditioned on the upper half (first 14 rows), the model generates a full 56-row sequence (2x the original 28 rows).

## `representation_metrics.json`
Scalar summary metrics for sequence evaluation, including validation prediction loss and latent-step geometry.
"""
    (out_dir / "plot_descriptions.md").write_text(md, encoding="utf-8")


def sequence_latent_step_stats(reps_seq: np.ndarray) -> dict:
    # reps_seq: [B, S, D]
    steps = reps_seq[:, 1:, :] - reps_seq[:, :-1, :]
    step_norm = np.linalg.norm(steps, axis=-1)
    return {
        "mean_step_l2": float(step_norm.mean()),
        "std_step_l2": float(step_norm.std()),
    }


def plot_sequence_latent_pca_samples(reps_seq: np.ndarray, labels: np.ndarray, num_classes: int, out_path: Path):
    b, s, d = reps_seq.shape
    flat = reps_seq.reshape(b * s, d)
    flat_labels = np.repeat(labels, s)
    z2d = pca_2d(flat)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(z2d[:, 0], z2d[:, 1], c=flat_labels, s=4, cmap="tab10", alpha=0.55, linewidths=0)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(False)
    fig.colorbar(sc, ax=ax, ticks=range(num_classes), label="Digit class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def project_sequence_latents_2d(reps_seq: np.ndarray, prefer: str = "tsne") -> np.ndarray:
    b, s, d = reps_seq.shape
    flat = reps_seq.reshape(b * s, d)
    if prefer == "tsne":
        try:
            z2d = tsne_2d(flat, random_state=42)
        except Exception:
            z2d = pca_2d(flat)
    else:
        z2d = pca_2d(flat)
    return z2d.reshape(b, s, 2)


def downsample_sequence_points(
    z2d: np.ndarray,
    labels: np.ndarray,
    max_points: int = 60000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    flat = z2d.reshape(-1, 2)
    flat_labels = np.repeat(labels, z2d.shape[1])
    n = flat.shape[0]
    if n <= max_points:
        return flat, flat_labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return flat[idx], flat_labels[idx]


def smooth_path(points: np.ndarray, interp_factor: int = 6, smooth_window: int = 7) -> np.ndarray:
    # points: [S, 2]
    s = points.shape[0]
    if s < 3:
        return points
    t = np.arange(s, dtype=np.float64)
    t_hi = np.linspace(0.0, float(s - 1), num=max(s * interp_factor, s), dtype=np.float64)
    x_hi = np.interp(t_hi, t, points[:, 0])
    y_hi = np.interp(t_hi, t, points[:, 1])
    if smooth_window > 1:
        k = np.ones(smooth_window, dtype=np.float64) / smooth_window
        x_hi = np.convolve(x_hi, k, mode="same")
        y_hi = np.convolve(y_hi, k, mode="same")
    return np.stack([x_hi, y_hi], axis=1)


def plot_sequence_mean_paths(
    z2d: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    out_path: Path,
):
    palette = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Global mean path.
    global_raw = z2d.mean(axis=0)
    global_smooth = smooth_path(global_raw, interp_factor=8, smooth_window=9)
    ax.plot(
        global_smooth[:, 0],
        global_smooth[:, 1],
        color="black",
        alpha=0.95,
        linewidth=1.6,
        linestyle="--",
        label="global mean",
    )
    ax.scatter(global_raw[:, 0], global_raw[:, 1], color="black", s=9, alpha=0.8)

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        color = palette(c % 10)
        raw_path = z2d[idx].mean(axis=0)
        smooth = smooth_path(raw_path, interp_factor=8, smooth_window=9)
        ax.plot(smooth[:, 0], smooth[:, 1], color=color, alpha=0.95, linewidth=1.2, label=f"{c}")
        # Mark original timestep points used to define path.
        ax.scatter(raw_path[:, 0], raw_path[:, 1], color=color, s=8, alpha=0.8)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(False)
    style_axes(ax, frameless=False)
    ax.legend(title="Digit", ncols=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_sequence_per_digit_paths(
    z2d: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    out_path: Path,
    max_paths_per_class: int = 20,
):
    palette = plt.get_cmap("tab10")
    rng = np.random.default_rng(42)
    ncols = 5
    nrows = int(np.ceil(num_classes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.8 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    for c in range(num_classes):
        ax = axes[c]
        idx = np.where(labels == c)[0]
        ax.grid(False)
        ax.set_title(f"Digit {c}", fontsize=8)
        style_axes(ax, frameless=True)
        if idx.size == 0:
            ax.axis("off")
            continue
        if idx.size > max_paths_per_class:
            idx = rng.choice(idx, size=max_paths_per_class, replace=False)

        color = palette(c % 10)
        class_paths = z2d[idx]  # [Nc, S, 2]
        for path in class_paths:
            path_smooth = smooth_path(path, interp_factor=6, smooth_window=7)
            ax.plot(path_smooth[:, 0], path_smooth[:, 1], color=color, alpha=0.18, linewidth=0.45)

        mean_raw = class_paths.mean(axis=0)
        mean_smooth = smooth_path(mean_raw, interp_factor=8, smooth_window=9)
        ax.plot(mean_smooth[:, 0], mean_smooth[:, 1], color=color, alpha=0.95, linewidth=1.2)
        ax.scatter(mean_raw[:, 0], mean_raw[:, 1], color=color, s=10, alpha=0.9)

    for i in range(num_classes, len(axes)):
        axes[i].axis("off")

    fig.supxlabel("t-SNE 1")
    fig.supylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _apply_encoder_layers_once(encoder_module: TRSeqEncoder | TRSeqElementwiseEncoder, step_input: torch.Tensor) -> torch.Tensor:
    cur = step_input
    for _pass_i, _unique_i, layer in encoder_module.enumerate_pass_layers():
        cur = layer(cur)
    return cur


def autoregressive_generate_rows(
    encoder_trainer: SeqEncoderTrainer,
    predictor: PredictorHead,
    x_seq: torch.Tensor,
    observed_rows: int = 14,
    total_rows: int | None = None,
) -> torch.Tensor:
    # x_seq: [B, S, D] normalized rows
    # returns generated sequence of length `total_rows` where rows < observed_rows are copied from input
    # and rows >= observed_rows are generated autoregressively.
    assert x_seq.ndim == 3
    b, s, d = x_seq.shape
    assert observed_rows >= 1 and observed_rows < s
    total_rows = s if total_rows is None else int(total_rows)
    assert total_rows >= observed_rows

    encoder_module = encoder_trainer.encoder
    device = x_seq.device
    out_seq = torch.zeros((b, total_rows, d), device=device, dtype=x_seq.dtype)
    out_seq[:, :observed_rows, :] = x_seq[:, :observed_rows, :]

    if isinstance(encoder_module, TRSeqEncoder):
        hidden = torch.zeros((b, encoder_module.rep_dim), device=device, dtype=x_seq.dtype)

        # Condition hidden on observed rows.
        for t in range(observed_rows):
            step_in = encoder_module.concat_x_t_hidden(x_seq, t, hidden)
            hidden = _apply_encoder_layers_once(encoder_module, step_in)

        cur_row = x_seq[:, observed_rows - 1, :]
        for t in range(observed_rows, total_rows):
            step_in = torch.cat([cur_row, hidden], dim=1)
            hidden = _apply_encoder_layers_once(encoder_module, step_in)
            pred_next = predictor.mapping(hidden)
            out_seq[:, t, :] = pred_next
            cur_row = pred_next
    else:
        # Elementwise sequence encoder: no recurrent hidden state.
        cur_row = x_seq[:, observed_rows - 1, :]
        for t in range(observed_rows, total_rows):
            hidden = _apply_encoder_layers_once(encoder_module, cur_row)
            pred_next = predictor.mapping(hidden)
            out_seq[:, t, :] = pred_next
            cur_row = pred_next

    return out_seq


def denorm_mnist_rows(x: np.ndarray) -> np.ndarray:
    # Reverse MNIST normalization used in dataset pipeline.
    mean = 0.1307
    std = 0.3081
    return np.clip(x * std + mean, 0.0, 1.0)


def plot_autoregressive_lower_half(
    encoder_trainer: SeqEncoderTrainer,
    predictor: PredictorHead,
    val_loader,
    out_path: Path,
    observed_rows: int = 14,
):
    x_all = []
    y_all = []
    with torch.no_grad():
        for x, y in val_loader:
            x_all.append(x)
            y_all.append(y)
    x_all = torch.cat(x_all, dim=0)
    y_all = torch.cat(y_all, dim=0)

    rng = np.random.default_rng(42)
    n = x_all.shape[0]
    pick = rng.choice(n, size=min(10, n), replace=False)
    x_pick = x_all[pick]
    y_pick = y_all[pick].numpy()

    with torch.no_grad():
        x_pred = autoregressive_generate_rows(
            encoder_trainer=encoder_trainer,
            predictor=predictor,
            x_seq=x_pick,
            observed_rows=observed_rows,
            total_rows=x_pick.shape[1],
        ).cpu().numpy()

    x_gt = x_pick.cpu().numpy()
    x_vis = denorm_mnist_rows(x_pred)
    x_gt_vis = denorm_mnist_rows(x_gt)

    fig, axes = plt.subplots(2, 5, figsize=(11, 4.8))
    axes = axes.reshape(-1)
    for i, ax in enumerate(axes):
        img = x_vis[i]
        mse = float(np.mean((x_pred[i, observed_rows:, :] - x_gt[i, observed_rows:, :]) ** 2))
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axhline(observed_rows - 0.5, color="tab:red", linewidth=0.7, linestyle="--")
        ax.set_title(f"d={int(y_pick[i])} mse={mse:.3f}", fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_autoregressive_double_rows(
    encoder_trainer: SeqEncoderTrainer,
    predictor: PredictorHead,
    val_loader,
    out_path: Path,
    observed_rows: int = 14,
):
    x_all = []
    y_all = []
    with torch.no_grad():
        for x, y in val_loader:
            x_all.append(x)
            y_all.append(y)
    x_all = torch.cat(x_all, dim=0)
    y_all = torch.cat(y_all, dim=0)

    rng = np.random.default_rng(123)
    n = x_all.shape[0]
    pick = rng.choice(n, size=min(10, n), replace=False)
    x_pick = x_all[pick]
    y_pick = y_all[pick].numpy()
    s = x_pick.shape[1]
    total_rows = 2 * s

    with torch.no_grad():
        x_gen = autoregressive_generate_rows(
            encoder_trainer=encoder_trainer,
            predictor=predictor,
            x_seq=x_pick,
            observed_rows=observed_rows,
            total_rows=total_rows,
        ).cpu().numpy()

    x_vis = denorm_mnist_rows(x_gen)

    fig, axes = plt.subplots(2, 5, figsize=(11, 8.2))
    axes = axes.reshape(-1)
    for i, ax in enumerate(axes):
        img = x_vis[i]
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axhline(observed_rows - 0.5, color="tab:red", linewidth=0.65, linestyle="--")
        ax.axhline(s - 0.5, color="tab:blue", linewidth=0.65, linestyle=":")
        ax.set_title(f"d={int(y_pick[i])}", fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--max_points", type=int, default=5000)
    parser.add_argument("--max_points_spectral", type=int, default=2000)
    parser.add_argument("--max_seq_samples", type=int, default=800)
    parser.add_argument("--max_seq_timesteps", type=int, default=0)
    parser.add_argument("--out_root", type=str, default="output/analysis")
    args = parser.parse_args()

    save_root = Path("saved_models")
    run_dir = Path(args.run_dir) if args.run_dir else newest_run_dir(save_root)
    out_dir = Path(args.out_root) / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(run_dir / "vicreg_hparams.pth")
    cfg.logger = "csv"
    cfg.store_config.device = "cpu"

    _train_loader, _head_train_loader, val_loader = build_dataloaders(cfg.data_config, cfg.problem_type)

    encoder_state = torch.load(run_dir / "vicreg_encoder.pth", map_location="cpu")
    # Older saved hparams may not include `encoders` (class var in Config). Infer from checkpoint.
    if "encoder.layers.0.lin.weight" in encoder_state:
        inferred = infer_encoders_from_state_dict(encoder_state)
        if tuple(cfg.encoders[0].layer_dims) != tuple(inferred[0].layer_dims):
            cfg.encoders = inferred
            finish_setup(cfg)

    encoder_cfg = cfg.encoders[0]
    encoder_cls = TREncoder
    if cfg.problem_type == "sequence":
        encoder_cls = TRSeqEncoder if cfg.sequence_recurrent_encoder else TRSeqElementwiseEncoder
    trainer_cls = SeqEncoderTrainer if cfg.problem_type == "sequence" else EncoderTrainer
    encoder = encoder_cls(cfg, encoder_cfg)
    encoder_trainer = trainer_cls("e0", cfg, encoder, pre_model=None)
    encoder_trainer.load_state_dict(encoder_state, strict=False)
    encoder_trainer.eval()

    head_cls = PredictorHead if cfg.problem_type == "sequence" else ClassifierHead
    clf = head_cls(encoder_trainer, cfg, cfg.head_out_dim)
    cls_path = run_dir / "vicreg_classifier.pth"
    if cls_path.exists():
        cls_state = torch.load(cls_path, map_location="cpu")
        clf.load_state_dict(cls_state, strict=False)
    clf.eval()

    reps = []
    labels_all = []
    imgs_all = []
    preds_all = []
    seq_pred_losses = []
    with torch.no_grad():
        for x, y in val_loader:
            r = encoder_trainer(x)
            if cfg.problem_type == "sequence":
                x_in = x[:, :-1, :]
                y_tgt = x[:, 1:, :]
                out = clf(x_in)
                seq_pred_losses.append(torch.mean((out - y_tgt) ** 2).item())
            else:
                out = clf(x)
                preds_all.append(out.argmax(dim=1).cpu())
            reps.append(r.cpu())
            labels_all.append(y.cpu())
            imgs_all.append(x.cpu())
    reps = torch.cat(reps, dim=0).numpy()
    labels = torch.cat(labels_all, dim=0).numpy()
    imgs = torch.cat(imgs_all, dim=0).numpy()
    preds = torch.cat(preds_all, dim=0).numpy() if preds_all else None

    if reps.shape[0] > args.max_points:
        idx = np.random.default_rng(42).choice(reps.shape[0], size=args.max_points, replace=False)
        reps_plot = reps[idx]
        labels_plot = labels[idx]
    else:
        reps_plot = reps
        labels_plot = labels

    setup_plot_style()

    if cfg.problem_type == "sequence":
        # Sequence-specific analysis for RNN-style models.
        reps_seq = reps
        labels_seq = labels

        # Keep sequence structure, but reduce t-SNE cost via sequence/timestep subsampling.
        b, s, _d = reps_seq.shape
        if b > args.max_seq_samples:
            idx_b = np.random.default_rng(42).choice(b, size=args.max_seq_samples, replace=False)
            reps_seq = reps_seq[idx_b]
            labels_seq = labels_seq[idx_b]
            b, s, _d = reps_seq.shape

        if args.max_seq_timesteps > 0 and s > args.max_seq_timesteps:
            t_idx = np.linspace(0, s - 1, num=args.max_seq_timesteps, dtype=int)
            reps_seq = reps_seq[:, t_idx, :]

        seq_z2d = project_sequence_latents_2d(reps_seq, prefer="tsne")
        np.save(out_dir / "sequence_latent_tsne_coords.npy", seq_z2d)
        np.save(out_dir / "sequence_latent_tsne_labels.npy", labels_seq)
        fig, ax = plt.subplots(figsize=(8, 6))
        flat_z, flat_labels = downsample_sequence_points(seq_z2d, labels_seq, max_points=60000, seed=42)
        sc = ax.scatter(flat_z[:, 0], flat_z[:, 1], c=flat_labels, s=3, cmap="tab10", alpha=0.38, linewidths=0)
        num_seq_classes = cfg.head_out_dim if cfg.head_task == "classification" else 10
        for c in range(num_seq_classes):
            class_idx = np.where(flat_labels == c)[0]
            if class_idx.size == 0:
                continue
            center = flat_z[class_idx].mean(axis=0)
            ax.scatter(center[0], center[1], marker="x", s=52, linewidths=0.9, color=plt.get_cmap("tab10")(c % 10))
            ax.text(center[0], center[1], str(c), fontsize=7, ha="center", va="center")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(False)
        style_axes(ax, frameless=False)
        fig.colorbar(sc, ax=ax, ticks=range(num_seq_classes), label="Digit class")
        fig.tight_layout()
        fig.savefig(out_dir / "sequence_latent_tsne_samples.png", dpi=240)
        plt.close(fig)
        plot_sequence_mean_paths(
            z2d=seq_z2d,
            labels=labels_seq,
            num_classes=num_seq_classes,
            out_path=out_dir / "sequence_latent_tsne_mean_paths.png",
        )
        plot_sequence_per_digit_paths(
            z2d=seq_z2d,
            labels=labels_seq,
            num_classes=num_seq_classes,
            out_path=out_dir / "sequence_latent_tsne_per_digit_paths.png",
        )
        plot_autoregressive_lower_half(
            encoder_trainer=encoder_trainer,
            predictor=clf,
            val_loader=val_loader,
            out_path=out_dir / "sequence_autoregressive_lower_half.png",
            observed_rows=14,
        )
        plot_autoregressive_double_rows(
            encoder_trainer=encoder_trainer,
            predictor=clf,
            val_loader=val_loader,
            out_path=out_dir / "sequence_autoregressive_double_rows.png",
            observed_rows=14,
        )

        step_stats = sequence_latent_step_stats(reps)
        metrics = {
            "val_prediction_loss": float(np.mean(seq_pred_losses)) if seq_pred_losses else float("nan"),
            **step_stats,
        }
        with (out_dir / "representation_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        write_sequence_plot_descriptions(out_dir)
        np.save(out_dir / "reps_val.npy", reps)
        np.save(out_dir / "labels_val.npy", labels)
        np.save(out_dir / "images_val.npy", imgs)
        print(f"Saved analysis to {out_dir}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    p2 = pca_2d(reps_plot)
    sc = scatter_latent(ax, p2, labels_plot, cfg.head_out_dim)
    fig.colorbar(sc, ax=ax, ticks=range(cfg.head_out_dim), label="Class")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_representations.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    lda2 = lda_2d(reps_plot, labels_plot, cfg.head_out_dim)
    sc = scatter_latent(ax, lda2, labels_plot, cfg.head_out_dim)
    fig.colorbar(sc, ax=ax, ticks=range(cfg.head_out_dim), label="Class")
    fig.tight_layout()
    fig.savefig(out_dir / "lda_representations.png", dpi=240)
    plt.close(fig)

    if reps_plot.shape[0] > args.max_points_spectral:
        sidx = np.random.default_rng(7).choice(reps_plot.shape[0], size=args.max_points_spectral, replace=False)
        reps_spec = reps_plot[sidx]
        labels_spec = labels_plot[sidx]
    else:
        reps_spec = reps_plot
        labels_spec = labels_plot

    fig, ax = plt.subplots(figsize=(8, 6))
    spec2 = spectral_2d(reps_spec, n_neighbors=12)
    sc = scatter_latent(ax, spec2, labels_spec, cfg.head_out_dim)
    fig.colorbar(sc, ax=ax, ticks=range(cfg.head_out_dim), label="Class")
    fig.tight_layout()
    fig.savefig(out_dir / "spectral_representations.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    tsne2 = tsne_2d(reps_spec, random_state=42)
    sc = scatter_latent(ax, tsne2, labels_spec, cfg.head_out_dim)
    fig.colorbar(sc, ax=ax, ticks=range(cfg.head_out_dim), label="Class")
    fig.tight_layout()
    fig.savefig(out_dir / "tsne_representations.png", dpi=240)
    plt.close(fig)

    first_w = encoder_trainer.encoder.layers[0].lin.weight.detach().cpu().numpy()
    l2 = np.linalg.norm(first_w, axis=1)
    top_idx = np.argsort(l2)[-64:][::-1]
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    robust = np.percentile(np.abs(first_w[top_idx]), 96) + 1e-8
    for i, ax in enumerate(axes.flat):
        w = first_w[top_idx[i]].reshape(28, 28)
        w_scaled = np.tanh(2.5 * w / robust)
        ax.imshow(w_scaled, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "first_layer_top64_rf.png", dpi=240)
    plt.close(fig)

    class_means = np.stack([reps[labels == c].mean(axis=0) for c in range(cfg.head_out_dim)], axis=0)
    selectivity = class_means.max(axis=0) - class_means.min(axis=0)
    top_sel = np.argsort(selectivity)[-48:][::-1]
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(class_means[:, top_sel], aspect="auto", cmap="viridis")
    ax.set_xlabel("Neuron (sorted by selectivity)")
    ax.set_ylabel("Class")
    ax.set_yticks(range(cfg.head_out_dim))
    ax.grid(False)
    style_axes(ax, frameless=False)
    fig.colorbar(im, ax=ax, label="Mean activation")
    fig.tight_layout()
    fig.savefig(out_dir / "top_selective_neurons_heatmap.png", dpi=240)
    plt.close(fig)

    conf = np.zeros((cfg.head_out_dim, cfg.head_out_dim), dtype=np.float64)
    for t, p in zip(labels, preds):
        conf[t, p] += 1.0
    conf_norm = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf_norm, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(cfg.head_out_dim))
    ax.set_yticks(range(cfg.head_out_dim))
    ax.grid(False)
    style_axes(ax, frameless=False)
    fig.colorbar(im, ax=ax, label="Row-normalized probability")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_normalized.png", dpi=240)
    plt.close(fig)

    proto = cosine_normalize(class_means.copy())
    proto_cos = proto @ proto.T
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(proto_cos, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xlabel("Class")
    ax.set_ylabel("Class")
    ax.set_xticks(range(cfg.head_out_dim))
    ax.set_yticks(range(cfg.head_out_dim))
    ax.grid(False)
    style_axes(ax, frameless=False)
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(out_dir / "class_prototype_cosine.png", dpi=240)
    plt.close(fig)

    cos_stats = pairwise_cosine_stats(reps, labels)
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(-1.0, 1.0, 80)
    ax.hist(cos_stats["different"], bins=bins, alpha=0.65, density=True, label="different class")
    ax.hist(cos_stats["same"], bins=bins, alpha=0.65, density=True, label="same class")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.grid(False)
    style_axes(ax, frameless=False)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pairwise_cosine_distribution.png", dpi=240)
    plt.close(fig)

    nn_class = proto_cos.copy()
    np.fill_diagonal(nn_class, -2.0)
    nearest_other_mean = float(np.mean(nn_class.max(axis=1)))
    head_acc = float((preds == labels).mean())
    metrics = {
        "head_val_acc": head_acc,
        "pairwise_same_mean_cos": float(cos_stats["same"].mean()),
        "pairwise_diff_mean_cos": float(cos_stats["different"].mean()),
        "pairwise_margin_cos": float(cos_stats["same"].mean() - cos_stats["different"].mean()),
        "prototype_self_mean_cos": float(np.mean(np.diag(proto_cos))),
        "prototype_nearest_other_mean_cos": nearest_other_mean,
        "mean_neuron_selectivity": float(selectivity.mean()),
        "top10pct_neuron_selectivity_mean": float(np.mean(np.sort(selectivity)[-max(1, len(selectivity) // 10):])),
    }
    with (out_dir / "representation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    write_plot_descriptions(out_dir)
    np.save(out_dir / "reps_val.npy", reps)
    np.save(out_dir / "labels_val.npy", labels)
    np.save(out_dir / "images_val.npy", imgs)
    print(f"Saved analysis to {out_dir}")


if __name__ == "__main__":
    main()
