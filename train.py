from trl import run_training
from trl.config.configurations import *

if __name__ == "__main__":
    cfg = mnist_backprop_comparison()
    cfg = beneficial_setup(cfg)
    cfg.trloss_config.std_coeff = 10.0
    cfg.trloss_config.cov_coeff = 10.0
    cfg.trloss_config.sim_coeff = 10.0
    cfg.trloss_config.detach_previous = True
    # cfg = ff_scale_network(cfg)
    # cfg.trloss_config.cov_matrix_sparsity = 0.75
    cfg = intermediate_length_run(cfg) 

    run_training.run(cfg)
