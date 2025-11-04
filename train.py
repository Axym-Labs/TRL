
import torch.nn as nn

from trl import run_training

if __name__ == "__main__":
    cfg = run_training.cfg
    cfg.encoders[0].activaton_fn = nn.GELU
    run_training.run(run_training.cfg)
