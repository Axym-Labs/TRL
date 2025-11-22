import torch
import torch.nn.functional as F

from trl import run_training
from trl.config.configurations import *

if __name__ == "__main__":
    cfg = intermediate_length_run()

    # cfg = Config()
    cfg.store_config.device = "cpu"
    cfg.trloss_config.variance_hinge_fn = F.softplus

    # cfg = intermediate_length_run(cfg)

    # eqprop_scale_network(cfg)
    # cfg = ff_scale_network(cfg)

    standard_setup(cfg)

    # cfg = aug_and_rbn_setup(cfg)

    finish_setup(cfg)

    run_training.run(cfg)
