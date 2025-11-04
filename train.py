
import torch.nn as nn

from trl import run_training
from trl.config.configurations import mnist_rnn_setup

if __name__ == "__main__":
    cfg = mnist_rnn_setup()
    print(cfg)
    run_training.run(cfg)
