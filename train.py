from trl import run_training
from trl.config.configurations import *

if __name__ == "__main__":
    # cfg = mnist_rnn_setup()
    cfg = mnist_backprop_comparison()
    cfg = beneficial_setup(cfg)
    cfg = ff_scale_network(cfg)

    run_training.run(cfg)
