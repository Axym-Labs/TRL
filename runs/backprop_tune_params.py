
import math
import random
from copy import deepcopy
from pprint import pprint
import traceback

import numpy as np
from tqdm import trange

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from comparison.backprop_mnist import run as run_backprop_mnist
from trl.config.configurations import mnist_backprop_comparison_tuning
from runs.trl_tune_params import N_TRIALS


def sample_lr(low=1e-5, high=1e-1):
    return float(np.exp(np.random.uniform(math.log(low), math.log(high))))

def main():
    best = {"lr": None, "val_acc": -float("inf")}
    for i in trange(N_TRIALS, desc="lr-random-search"):
        lr = sample_lr()
        cfg = mnist_backprop_comparison_tuning()
        epochs = cfg.epochs
        batch_size = cfg.data_config.batch_size
        batchnorm = (cfg.batchnorm_config is not None)

        print(f"\nTrial {i+1}/{N_TRIALS} — lr={lr:.3g}")
        pprint(dict(lr=lr, epochs=epochs, batch_size=batch_size, batchnorm=batchnorm))

        try:
            val_acc = run_backprop_mnist(epochs=epochs, batch_size=batch_size, batchnorm=batchnorm, lr=lr)
        except Exception as e:
            print(f"TRIAL FAILED")
            traceback.print_exc()
            val_acc = None

        print(f"Result val_acc = {val_acc}")

        if val_acc is not None and val_acc > best["val_acc"]:
            best = {"lr": lr, "val_acc": val_acc}

    print("\n=== Best LR ===")
    pprint(best)

if __name__ == "__main__":
    main()
