import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from analysis.model_representation_analysis import (
    downsample_sequence_points,
    project_sequence_latents_2d,
    setup_plot_style,
    style_axes,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_dir", type=str, required=True)
    parser.add_argument("--max_seq_samples", type=int, default=5000)
    parser.add_argument("--max_seq_timesteps", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=60000)
    parser.add_argument("--out_name", type=str, default="sequence_latent_tsne_samples.png")
    parser.add_argument("--compute_if_missing", action="store_true")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    reps_path = analysis_dir / "reps_val.npy"
    labels_path = analysis_dir / "labels_val.npy"
    coords_path = analysis_dir / "sequence_latent_tsne_coords.npy"
    coords_labels_path = analysis_dir / "sequence_latent_tsne_labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Missing cached arrays in {analysis_dir}. Expected labels_val.npy."
        )
    labels = np.load(labels_path)

    if coords_path.exists() and coords_labels_path.exists():
        seq_z2d = np.load(coords_path)
        labels = np.load(coords_labels_path)
        if seq_z2d.ndim != 3 or seq_z2d.shape[0] != labels.shape[0]:
            raise ValueError(f"Invalid cached coords shape {seq_z2d.shape} for labels shape {labels.shape}")
    else:
        if not args.compute_if_missing:
            raise FileNotFoundError(
                f"Missing cached coordinates in {analysis_dir}. Run once with --compute_if_missing."
            )
        if not reps_path.exists():
            raise FileNotFoundError(f"Missing {reps_path}; cannot compute t-SNE coordinates.")
        reps = np.load(reps_path)
        if reps.ndim != 3:
            raise ValueError(f"Expected sequence reps with shape [B,S,D], got shape {reps.shape}")
        if reps.shape[0] > args.max_seq_samples:
            idx = np.random.default_rng(42).choice(reps.shape[0], size=args.max_seq_samples, replace=False)
            reps = reps[idx]
            labels = labels[idx]
        if args.max_seq_timesteps > 0 and reps.shape[1] > args.max_seq_timesteps:
            t_idx = np.linspace(0, reps.shape[1] - 1, num=args.max_seq_timesteps, dtype=int)
            reps = reps[:, t_idx, :]
        seq_z2d = project_sequence_latents_2d(reps, prefer="tsne")
        np.save(coords_path, seq_z2d)
        np.save(coords_labels_path, labels)

    setup_plot_style()
    flat_z, flat_labels = downsample_sequence_points(seq_z2d, labels, max_points=args.max_points, seed=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(flat_z[:, 0], flat_z[:, 1], c=flat_labels, s=3, cmap="tab10", alpha=0.38, linewidths=0)
    for c in np.unique(flat_labels):
        class_idx = np.where(flat_labels == c)[0]
        center = flat_z[class_idx].mean(axis=0)
        ax.scatter(center[0], center[1], marker="x", s=52, linewidths=0.9, color=plt.get_cmap("tab10")(int(c) % 10))
        ax.text(center[0], center[1], str(int(c)), fontsize=7, ha="center", va="center")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(False)
    style_axes(ax, frameless=False)
    fig.colorbar(sc, ax=ax, ticks=range(int(flat_labels.max()) + 1), label="Digit class")
    fig.tight_layout()
    out_path = analysis_dir / args.out_name
    fig.savefig(out_path, dpi=240)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
