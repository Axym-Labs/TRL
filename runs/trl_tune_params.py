
import random
import math
from copy import deepcopy
from pprint import pprint
import traceback

import numpy as np
from tqdm import trange

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trl import run_training
from trl.config.configurations import mnist_backprop_comparison_tuning

N_TRIALS = 50

def sample_trial():
    lr = float(np.exp(np.random.uniform(math.log(1e-5), math.log(1e-1))))
    # sample std first as reference scale
    std = float(np.exp(np.random.uniform(math.log(2), math.log(120))))

    # sim < std, cov < std
    sim = float(np.random.uniform(1, std))
    cov_coeff = float(np.random.uniform(1, std))
    lat = cov_coeff # tie lat and cov

    return dict(
        lr=lr,
        sim_coeff=sim,
        std_coeff=std,
        lat_coeff=lat,
        cov_coeff=cov_coeff,
    )

def apply_to_cfg(cfg, s):
    cfg = deepcopy(cfg)
    cfg.lr = s["lr"]
    cfg.tcloss_config.sim_coeff = s["sim_coeff"]
    cfg.tcloss_config.std_coeff = s["std_coeff"]
    cfg.tcloss_config.lat_coeff = s["lat_coeff"]
    cfg.tcloss_config.cov_coeff = s["cov_coeff"]
    return cfg

def main():
    best_val, best_trial = -float("inf"), None

    for i in trange(N_TRIALS, desc="random-search"):
        s = sample_trial()
        cfg = apply_to_cfg(mnist_backprop_comparison_tuning(), s)

        print(f"\nTrial {i+1}/{N_TRIALS} params:")
        pprint(s)

        try:
            val_acc = run_training.run(cfg)
        except Exception as e:
            print("TRIAL FAILED")
            traceback.print_exc()
            val_acc = None

        print(f"Trial {i+1} result — val_acc: {val_acc}")

        if val_acc is not None and val_acc > best_val:
            best_val = val_acc
            best_trial = dict(params=s, val_acc=val_acc)

    print("\n=== Best trial ===")
    pprint(best_trial)


if __name__ == "__main__":
    main()
