import numpy as np
import traceback
from copy import deepcopy

from trl import run_training
from trl.config.configurations import *



def run_with_seeds(cfg):
    finish_setup(cfg)
    
    accs = []
    seeds = [42, 43, 44, 45, 46]

    for seed in seeds:
        cfg_current = deepcopy(cfg)
        cfg_current.seed = seed
        try:
            val_acc = run_training.run(cfg_current)
            accs.append(val_acc)
        except Exception as e:
            print(f"Run with seed {seed} failed:")
            traceback.print_exc()

    print("Runs ", accs)
    print(f"[INCOMPLETE]: Average val acc over seeds: {np.mean(accs):.4f} ± {np.std(accs):.4f}, val error: {1 - np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print("---")
        
def print_section(s):
    print("="*40)
    print(s)
    print("="*40)

if __name__ == "__main__":
    print("Test")
    cfg = Config()
    cfg.encoders[0].layer_dims = ((28**2, 256),)
    cfg.epochs = 1
    cfg.head_epochs = 1
    run_with_seeds(cfg)

    print_section("512->256 TRL")
    run_with_seeds(standard_setup(long_training()))

    print_section("\n512->256 TRL-S")
    run_with_seeds(long_training())

    print_section("512->256 TRL AUG & BN")
    run_with_seeds(aug_and_rbn_setup(standard_setup(long_training())))
    
    print_section("\n512->256 TRL-S AUG & BN")
    run_with_seeds(aug_and_rbn_setup(long_training()))

    print_section("\nEqprop Scale TRL")
    run_with_seeds(eqprop_scale_network(standard_setup(long_training())))

    print_section("\nEqProp Scale TRL-S")
    run_with_seeds(eqprop_scale_network(long_training()))

    print_section("\nFF Scale TRL")
    run_with_seeds(ff_scale_network(standard_setup(long_training())))

    print_section("\nFF Scale TRL-S")
    run_with_seeds(ff_scale_network(long_training()))

