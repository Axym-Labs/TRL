import numpy as np
import traceback
from copy import deepcopy
import logging

from trl import run_training
from trl.config.configurations import *

logger = logging.getLogger(__name__)
logging.basicConfig(filename="all_comparison_runs_out.log",  level=logging.INFO)

def run_with_seeds(cfg, seed_start_at=42, num_runs=5):
    finish_setup(cfg)
    
    accs = []
    seeds = list(range(seed_start_at,seed_start_at+num_runs))

    for seed in seeds:
        cfg_current = deepcopy(cfg)
        cfg_current.seed = seed
        try:
            val_acc = run_training.run(cfg_current)
            accs.append(val_acc)
        except Exception as e:
            print(f"Run with seed {seed} failed:")
            traceback.print_exc()

    logger.info(f"Runs: {accs}")
    logger.info(f"Average val acc over seeds: {np.mean(accs):.4f} ± {np.std(accs):.4f}, val error: {1 - np.mean(accs):.4f} ± {np.std(accs):.4f}")
    logger.info("---")
        
def print_section(s):
    logger.info("="*40)
    logger.info(s)

if __name__ == "__main__":
    def base_cfg():
        cfg = Config()
        cfg.store_config.device = "cpu"
        return cfg

    # print_section("Test")
    # cfg = base_cfg()
    # cfg.encoders[0].layer_dims = ((28**2, 256),)
    # cfg.epochs = 1
    # cfg.head_epochs = 1
    # run_with_seeds(cfg, num_runs=1)

    print_section("512->256 TRL")
    run_with_seeds(standard_setup(long_training(base_cfg())), seed_start_at=43, num_runs=4)

    print_section("\n512->256 TRL-S")
    run_with_seeds(long_training(base_cfg()), seed_start_at=43, num_runs=4)

    print_section("512->256 TRL AUG & BN")
    run_with_seeds(aug_and_rbn_setup(standard_setup(long_training(base_cfg()))))
    
    print_section("\n512->256 TRL-S AUG & BN")
    run_with_seeds(aug_and_rbn_setup(long_training(base_cfg())))

    print_section("\nEqprop Scale TRL")
    run_with_seeds(eqprop_scale_network(standard_setup(long_training(base_cfg()))))

    print_section("\nEqProp Scale TRL-S")
    run_with_seeds(eqprop_scale_network(long_training(base_cfg())))

    print_section("\nFF Scale TRL")
    run_with_seeds(ff_scale_network(standard_setup(long_training(base_cfg()))))

    print_section("\nFF Scale TRL-S")
    run_with_seeds(ff_scale_network(long_training(base_cfg())))

