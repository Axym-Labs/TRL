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
from trl.modules.encoder import TREncoder, TRSeqEncoder
from trl.trainer.encoder import EncoderTrainer, SeqEncoderTrainer
from trl.trainer.head import ClassifierHead


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
    })


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
    except Exception as exc:
        raise RuntimeError("t-SNE requires scikit-learn. Install with `uv pip install scikit-learn`.") from exc

    perplexity = max(5, min(30, (x.shape[0] - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--max_points", type=int, default=5000)
    parser.add_argument("--max_points_spectral", type=int, default=2000)
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

    encoder_cfg = cfg.encoders[0]
    encoder_cls = TRSeqEncoder if cfg.problem_type == "sequence" else TREncoder
    trainer_cls = SeqEncoderTrainer if cfg.problem_type == "sequence" else EncoderTrainer
    encoder = encoder_cls(cfg, encoder_cfg)
    encoder_trainer = trainer_cls("e0", cfg, encoder, pre_model=None)
    encoder_state = torch.load(run_dir / "vicreg_encoder.pth", map_location="cpu")
    encoder_trainer.load_state_dict(encoder_state, strict=False)
    encoder_trainer.eval()

    clf = ClassifierHead(encoder_trainer, cfg, cfg.head_out_dim)
    cls_path = run_dir / "vicreg_classifier.pth"
    if cls_path.exists():
        cls_state = torch.load(cls_path, map_location="cpu")
        clf.load_state_dict(cls_state, strict=False)
    clf.eval()

    reps = []
    labels_all = []
    imgs_all = []
    preds_all = []
    with torch.no_grad():
        for x, y in val_loader:
            r = encoder_trainer(x)
            out = clf(x)
            preds_all.append(out.argmax(dim=1).cpu())
            reps.append(r.cpu())
            labels_all.append(y.cpu())
            imgs_all.append(x.cpu())
    reps = torch.cat(reps, dim=0).numpy()
    labels = torch.cat(labels_all, dim=0).numpy()
    imgs = torch.cat(imgs_all, dim=0).squeeze(1).numpy()
    preds = torch.cat(preds_all, dim=0).numpy()

    if reps.shape[0] > args.max_points:
        idx = np.random.default_rng(42).choice(reps.shape[0], size=args.max_points, replace=False)
        reps_plot = reps[idx]
        labels_plot = labels[idx]
    else:
        reps_plot = reps
        labels_plot = labels

    setup_plot_style()

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
