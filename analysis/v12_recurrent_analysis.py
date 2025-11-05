# analyze_vicreg_reps_imports.py
import os
import math
import sys
import multiprocessing as mp
from pathlib import Path
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from trl.run_training import Config, EncoderConfig, EncoderTrainer, ClassifierHead, build_dataloaders


# paths
ENCODER_PATH  = "saved_models/vicreg_9_covar_coarse/vicreg_encoder.pth"
CLASSIFIER_PATH = "saved_models/vicreg_9_covar_coarse/vicreg_classifier.pth"
OUT_DIR = Path("analysis_outputs")
(OUT_DIR / "first_layer_rf").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "second_layer_top6").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "second_layer_dists").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "second_layer_activations").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "classifier_analysis").mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unnormalize(img_tensor, mean=0.1307, std=0.3081):
    return img_tensor * std + mean

def analyze(encoder_path=ENCODER_PATH, classifier_path=CLASSIFIER_PATH):
    cfg = Config()
    encoder_cfg = EncoderConfig(layer_dims=((28*28, 512), (512, 256)))

    encoder_trainer = EncoderTrainer("e0", cfg, encoder_cfg, pre_model=None)
    encoder_trainer.to(DEVICE)

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder weights not found at {encoder_path}")
    state = torch.load(encoder_path, map_location=DEVICE)
    encoder_trainer.load_state_dict(state, strict=False)

    encoder_trainer.eval()

    classifier_model = ClassifierHead(encoder_trainer, cfg, out_dim=10)
    if os.path.exists(classifier_path):
        cls_state = torch.load(classifier_path, map_location=DEVICE)
        classifier_model.load_state_dict(cls_state, strict=False)

    classifier_model.to(DEVICE)
    classifier_model.eval()

    _, _, val_loader = build_dataloaders(cfg)

    all_acts = []
    all_labels = []
    all_images = []

    second_layer_idx = 1
    second_layer = encoder_trainer.encoder.layers[second_layer_idx]
    n_neurons_2 = second_layer.lin.out_features

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="collecting activations")):
            imgs, labels = batch
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            inp_to_layer2 = encoder_trainer.encoder.acts_before_layer(imgs, layer_idx=second_layer_idx, no_grad=True)
            acts = second_layer.forward(inp_to_layer2)
            all_acts.append(acts.detach().cpu())
            all_labels.append(labels.cpu())
            imgs_unnorm = unnormalize(imgs.cpu()).squeeze(1)
            all_images.append(imgs_unnorm)

    all_acts = torch.cat(all_acts, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_images = torch.cat(all_images, dim=0).numpy()

    np.save(OUT_DIR / "second_layer_activations" / "activations_all.npy", all_acts)
    np.save(OUT_DIR / "second_layer_activations" / "labels_all.npy", all_labels)

    first_lin = encoder_trainer.encoder.layers[0].lin
    W1 = first_lin.weight.detach().cpu().numpy()
    n_neurons_1 = W1.shape[0]

    for idx in range(n_neurons_1):
        w = W1[idx].reshape(28, 28)
        mn, mx = w.min(), w.max()
        img = np.zeros_like(w) if mx - mn == 0 else (w - mn) / (mx - mn)
        plt.figure(figsize=(3,3))
        plt.imshow(img, cmap='seismic', vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(OUT_DIR / "first_layer_rf" / f"layer1_neuron_{idx:03d}.png", bbox_inches='tight', pad_inches=0.05)
        plt.close()

    for neuron_idx in range(n_neurons_2):
        acts_vec = all_acts[:, neuron_idx]
        np.save(OUT_DIR / "second_layer_activations" / f"neuron_{neuron_idx:03d}_activations.npy", acts_vec)
        np.save(OUT_DIR / "second_layer_activations" / f"neuron_{neuron_idx:03d}_labels.npy", all_labels)

        topk = 6
        topk_idx = np.argsort(acts_vec)[-topk:][::-1]
        fig, axes = plt.subplots(2, 3, figsize=(6,4))
        fig.suptitle(f"Neuron {neuron_idx} top {topk}")
        for i, ax in enumerate(axes.flat):
            if i < len(topk_idx):
                idx = topk_idx[i]
                ax.imshow(all_images[idx], cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"{int(all_labels[idx])} / {acts_vec[idx]:.3f}")
            ax.axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(OUT_DIR / "second_layer_top6" / f"neuron_{neuron_idx:03d}_top6.png", dpi=150)
        plt.close()

        per_class_acts = [acts_vec[all_labels == c] for c in range(10)]
        plt.figure(figsize=(8,4))
        plt.boxplot(per_class_acts, labels=[str(i) for i in range(10)], showfliers=False)
        plt.xlabel("class")
        plt.ylabel("activation")
        plt.savefig(OUT_DIR / "second_layer_dists" / f"neuron_{neuron_idx:03d}_per_class_box.png", dpi=150)
        plt.close()

    Wc = classifier_model.mapping.weight.detach().cpu().numpy()
    num_classes, rep_dim = Wc.shape
    plt.figure(figsize=(12,6))
    plt.imshow(Wc, aspect='auto', cmap='bwr')
    plt.colorbar(label='weight value')
    plt.xlabel("feature")
    plt.ylabel("class")
    plt.savefig(OUT_DIR / "classifier_analysis" / "classifier_weight_heatmap.png", dpi=180)
    plt.close()

    topk_feats = 8
    cols = 5
    rows = math.ceil(num_classes / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*2.5))
    axs = axs.flatten()
    for c in range(num_classes):
        ax = axs[c]
        weights = Wc[c]
        top_pos = np.argsort(weights)[-topk_feats:][::-1]
        top_neg = np.argsort(weights)[:topk_feats]
        ax.bar(np.arange(topk_feats), weights[top_pos], label='pos', alpha=0.8)
        ax.bar(np.arange(topk_feats)+0.35, weights[top_neg], label='neg', alpha=0.8)
        ax.set_title(f"class {c}")
        ax.set_xticks([])
    for i in range(num_classes, len(axs)):
        axs[i].axis('off')
    plt.suptitle("Per-class top features")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUT_DIR / "classifier_analysis" / "classifier_top_features_per_class.png", dpi=180)
    plt.close()

def main():
    analyze()

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
