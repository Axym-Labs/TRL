import torch

from trl import run_training
from trl.config.configurations import *

if __name__ == "__main__":
    cfg = long_training()
    # cfg = intermediate_length_run(cfg)
    cfg = standard_setup(cfg)
    cfg = aug_and_rbn_setup(cfg)

    # cfg = eqprop_scale_network(cfg)

    # cfg = ff_scale_network(cfg)
    # 

    run_training.run(cfg)
