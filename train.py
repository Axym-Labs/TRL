from trl import run_training
from trl.config.configurations import *

if __name__ == "__main__":
    # cfg = mnist_rnn_setup()
    cfg = mnist_backprop_comparison_tuning()
    run_training.run(cfg)
