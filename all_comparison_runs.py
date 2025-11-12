import numpy as np
import traceback
from copy import deepcopy

from trl import run_training
from trl.config.configurations import *



def run_with_seeds(cfg, five_runs=False):
    accs = []
    seeds = [43, 44, 45, 46]
    if five_runs:
        seeds.insert(0, 42)

    for seed in [43, 44, 45, 46]:
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
        

if __name__ == "__main__":
    print("Test")
    cfg = Config()
    cfg.encoders[0].layer_dims = ((28**2, 256),)
    cfg.epochs = 1
    cfg.head_epochs = 1
    run_with_seeds(cfg)

    print("512->256 TRL")
    run_with_seeds(standard_setup(long_training()))

    print("\n512->256 TRL-S")
    run_with_seeds(long_training())

    print("512->256 TRL AUG & BN")
    run_with_seeds(aug_and_rbn_setup(standard_setup(long_training())))
    
    print("\n512->256 TRL-S AUG & BN")
    run_with_seeds(aug_and_rbn_setup(long_training()))

    print("\nEqprop Scale TRL")
    run_with_seeds(eqprop_scale_network(standard_setup(long_training())))

    print("\nEqProp Scale TRL-S")
    run_with_seeds(eqprop_scale_network(long_training()))

    print("\nFF Scale TRL")
    run_with_seeds(ff_scale_network(standard_setup(long_training())), five_runs=True)

    print("\nFF Scale TRL-S")
    run_with_seeds(ff_scale_network(long_training()), five_runs=True)

