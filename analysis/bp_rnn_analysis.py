import argparse
import json
from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trl.config.config import Config
from trl.config.configurations import finish_setup, mnist_rnn_setup
from trl.datasets.mnist import build_dataloaders
from trl.run_backprop import BPSeqLinearPredictor, BPSeqMLPPredictor


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
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, :2] * s[:2]


def tsne_2d(x: np.ndarray, random_state: int = 42) -> np.ndarray:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x = x / np.maximum(x_std, 1e-8)

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


def smooth_path(points: np.ndarray, interp_factor: int = 6, smooth_window: int = 7) -> np.ndarray:
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


def project_sequence_latents_2d(reps_seq: np.ndarray) -> np.ndarray:
    b, s, d = reps_seq.shape
    flat = reps_seq.reshape(b * s, d)
    try:
        z2d = tsne_2d(flat, random_state=42)
    except Exception:
        z2d = pca_2d(flat)
    return z2d.reshape(b, s, 2)


def downsample_sequence_points(z2d: np.ndarray, labels: np.ndarray, max_points: int = 60000, seed: int = 42):
    flat = z2d.reshape(-1, 2)
    flat_labels = np.repeat(labels, z2d.shape[1])
    n = flat.shape[0]
    if n <= max_points:
        return flat, flat_labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return flat[idx], flat_labels[idx]


def extract_seq_reps(model: BPSeqMLPPredictor | BPSeqLinearPredictor, x: torch.Tensor) -> torch.Tensor:
    # Returns hidden sequence of shape [B, S, H]
    bsz, seq_len, _ = x.shape
    h = x.new_zeros((bsz, model.hidden_dim))
    reps = []
    for t in range(seq_len):
        cur = torch.cat([x[:, t, :], h], dim=1)
        for layer in model.layers:
            cur = layer(cur)
            if isinstance(model, BPSeqMLPPredictor):
                cur = model.act(cur)
        h = cur
        reps.append(cur)
    return torch.stack(reps, dim=1)


def bp_autoregressive_generate(
    model: BPSeqMLPPredictor | BPSeqLinearPredictor,
    x_seq: torch.Tensor,
    observed_rows: int,
    total_rows: int,
) -> torch.Tensor:
    b, s, d = x_seq.shape
    out = torch.zeros((b, total_rows, d), device=x_seq.device, dtype=x_seq.dtype)
    out[:, :observed_rows, :] = x_seq[:, :observed_rows, :]
    h = x_seq.new_zeros((b, model.hidden_dim))

    for t in range(observed_rows):
        cur = torch.cat([x_seq[:, t, :], h], dim=1)
        for layer in model.layers:
            cur = layer(cur)
            if isinstance(model, BPSeqMLPPredictor):
                cur = model.act(cur)
        h = cur

    cur_row = x_seq[:, observed_rows - 1, :]
    for t in range(observed_rows, total_rows):
        cur = torch.cat([cur_row, h], dim=1)
        for layer in model.layers:
            cur = layer(cur)
            if isinstance(model, BPSeqMLPPredictor):
                cur = model.act(cur)
        h = cur
        nxt = model.head(cur)
        out[:, t, :] = nxt
        cur_row = nxt

    return out


def denorm_mnist_rows(x: np.ndarray) -> np.ndarray:
    mean = 0.1307
    std = 0.3081
    return np.clip(x * std + mean, 0.0, 1.0)


def plot_sequence_mean_paths(z2d: np.ndarray, labels: np.ndarray, out_path: Path, num_classes: int = 10):
    palette = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 6))

    global_raw = z2d.mean(axis=0)
    global_smooth = smooth_path(global_raw, interp_factor=8, smooth_window=9)
    ax.plot(global_smooth[:, 0], global_smooth[:, 1], color="black", alpha=0.95, linewidth=1.6, linestyle="--", label="global mean")
    ax.scatter(global_raw[:, 0], global_raw[:, 1], color="black", s=9, alpha=0.8)

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        color = palette(c % 10)
        raw_path = z2d[idx].mean(axis=0)
        smooth = smooth_path(raw_path, interp_factor=8, smooth_window=9)
        ax.plot(smooth[:, 0], smooth[:, 1], color=color, alpha=0.95, linewidth=1.2, label=f"{c}")
        ax.scatter(raw_path[:, 0], raw_path[:, 1], color=color, s=8, alpha=0.8)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(False)
    style_axes(ax, frameless=False)
    ax.legend(title="Digit", ncols=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_sequence_per_digit_paths(z2d: np.ndarray, labels: np.ndarray, out_path: Path, num_classes: int = 10, max_paths_per_class: int = 20):
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
        class_paths = z2d[idx]
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


def plot_autoregressive_grid(images: np.ndarray, labels: np.ndarray, out_path: Path, observed_rows: int, first_length: int, title_mse: np.ndarray | None = None):
    fig, axes = plt.subplots(2, 5, figsize=(11, 7.0 if images.shape[1] > first_length else 4.8))
    axes = axes.reshape(-1)
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap="gray", vmin=0.0, vmax=1.0)
        ax.axhline(observed_rows - 0.5, color="tab:red", linewidth=0.65, linestyle="--")
        if images.shape[1] > first_length:
            ax.axhline(first_length - 0.5, color="tab:blue", linewidth=0.65, linestyle=":")
        t = f"d={int(labels[i])}"
        if title_mse is not None:
            t += f" mse={title_mse[i]:.3f}"
        ax.set_title(t, fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_samples", type=int, default=800)
    parser.add_argument("--out_dir", type=str, default="output/analysis/bp_rnn_mnist_rows")
    parser.add_argument("--linear", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    cfg = Config()
    cfg = mnist_rnn_setup(cfg)
    cfg = finish_setup(cfg)
    cfg.logger = "csv"
    cfg.data_config.num_workers = 0
    cfg.data_config.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.head_epochs = args.epochs
    cfg.run_name = "bp_rnn_sequence_analysis"
    cfg.bp_sequence_linear = bool(args.linear)

    pl.seed_everything(cfg.seed)
    train_loader, _, val_loader = build_dataloaders(cfg.data_config, cfg.problem_type)
    model = BPSeqLinearPredictor(cfg) if cfg.bp_sequence_linear else BPSeqMLPPredictor(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_loader, val_loader)
    val_results = trainer.validate(model, dataloaders=val_loader)
    val_results = val_results[0] if val_results else {}
    val_loss = float(val_results.get("val_prediction_loss", np.nan))

    reps_all = []
    labels_all = []
    x_all = []
    pred_losses = []
    with torch.no_grad():
        for x, y in val_loader:
            reps = extract_seq_reps(model, x[:, :-1, :])
            pred = model(x[:, :-1, :])
            tgt = x[:, 1:, :]
            pred_losses.append(torch.mean((pred - tgt) ** 2).item())
            reps_all.append(reps.cpu())
            labels_all.append(y.cpu())
            x_all.append(x.cpu())
    reps = torch.cat(reps_all, dim=0).numpy()
    labels = torch.cat(labels_all, dim=0).numpy()
    x_full = torch.cat(x_all, dim=0)

    if reps.shape[0] > args.max_seq_samples:
        idx = np.random.default_rng(42).choice(reps.shape[0], size=args.max_seq_samples, replace=False)
        reps = reps[idx]
        labels = labels[idx]

    z2d = project_sequence_latents_2d(reps)
    np.save(out_dir / "sequence_latent_tsne_coords.npy", z2d)
    np.save(out_dir / "sequence_latent_tsne_labels.npy", labels)
    flat_z, flat_labels = downsample_sequence_points(z2d, labels, max_points=60000, seed=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(flat_z[:, 0], flat_z[:, 1], c=flat_labels, s=3, cmap="tab10", alpha=0.38, linewidths=0)
    for c in range(10):
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
    fig.colorbar(sc, ax=ax, ticks=range(10), label="Digit class")
    fig.tight_layout()
    fig.savefig(out_dir / "sequence_latent_tsne_samples.png", dpi=240)
    plt.close(fig)

    plot_sequence_mean_paths(z2d, labels, out_dir / "sequence_latent_tsne_mean_paths.png")
    plot_sequence_per_digit_paths(z2d, labels, out_dir / "sequence_latent_tsne_per_digit_paths.png")

    # Autoregressive 28-row and 56-row visuals.
    rng = np.random.default_rng(123)
    pick = rng.choice(x_full.shape[0], size=min(10, x_full.shape[0]), replace=False)
    x_pick = x_full[pick]
    y_pick = torch.cat(labels_all, dim=0)[pick].numpy()

    with torch.no_grad():
        pred_28 = bp_autoregressive_generate(model, x_pick, observed_rows=14, total_rows=28).cpu().numpy()
        pred_56 = bp_autoregressive_generate(model, x_pick, observed_rows=14, total_rows=56).cpu().numpy()
    gt_28 = x_pick.cpu().numpy()
    mse_28 = np.mean((pred_28[:, 14:, :] - gt_28[:, 14:, :]) ** 2, axis=(1, 2))
    plot_autoregressive_grid(
        images=denorm_mnist_rows(pred_28),
        labels=y_pick,
        out_path=out_dir / "sequence_autoregressive_lower_half.png",
        observed_rows=14,
        first_length=28,
        title_mse=mse_28,
    )
    plot_autoregressive_grid(
        images=denorm_mnist_rows(pred_56),
        labels=y_pick,
        out_path=out_dir / "sequence_autoregressive_double_rows.png",
        observed_rows=14,
        first_length=28,
        title_mse=None,
    )

    step = reps[:, 1:, :] - reps[:, :-1, :]
    step_norm = np.linalg.norm(step, axis=-1)
    metrics = {
        "val_prediction_loss": val_loss,
        "val_prediction_loss_manual_mean": float(np.mean(pred_losses)),
        "mean_step_l2": float(step_norm.mean()),
        "std_step_l2": float(step_norm.std()),
        "model_type": "bp_sequence_linear" if cfg.bp_sequence_linear else "bp_sequence_mlp",
        "epochs": int(cfg.epochs),
        "max_seq_samples": int(args.max_seq_samples),
    }
    (out_dir / "representation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.save(out_dir / "reps_val.npy", reps)
    np.save(out_dir / "labels_val.npy", labels)
    np.save(out_dir / "images_val.npy", x_full.numpy())
    print(f"Saved BP RNN analysis to {out_dir}")


if __name__ == "__main__":
    main()
