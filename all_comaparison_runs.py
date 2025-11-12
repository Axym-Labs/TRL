import numpy as np
import traceback

from trl import run_training
from trl.config.configurations import *



def run_with_seeds(cfg):
    accs = []
    for seed in [42, 43, 44, 45, 46]:
        cfg_current = deepcopy(cfg)
        cfg_current.seed = seed
        try:
            val_acc = run_training.run(cfg_current)
            accs.append(val_acc)
        except Exception as e:
            print(f"Run with seed {seed} failed:")
            traceback.print_exc()

    print("Runs ", accs)
    print(f"Average val acc over seeds: {np.mean(accs):.4f} ± {np.std(accs):.4f}, val error: {1 - np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print("---")
        

if __name__ == "__main__":
    print("Test")
    cfg = Config()
    cfg.encoders[0].layer_dims = [28**2, 256]
    cfg.epochs = 5
    cfg.head_epochs = 5
    run_with_seeds(cfg)

    print("512->256 TRL")
    # TODO
    run_with_seeds(cfg)

    print("512->256 TRL AUG & RBN")
    run_with_seeds(cfg)

    print("\n512->256 TRL-S")
    run_with_seeds(cfg)

    print("\n500->500->500 TRL")
    run_with_seeds(cfg)

    print("\n500->500->500 TRL-S")
    run_with_seeds(cfg)

    print("\nFF Scale Network TRL")
    run_with_seeds(cfg)

    print("\nFF Scale Network TRL-S")
    run_with_seeds(cfg)

