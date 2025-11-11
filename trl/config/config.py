from dataclasses import dataclass
from typing import Tuple
from torch import nn as nn

@dataclass
class BatchNormConfig:
    eps: float = 1e-5
    use_mean: bool = True
    use_variance: bool = True
    scale_parameter: bool = True
    bias_parameter: bool = True

    use_batch_statistics_training: bool = True
    detach_batch_statistics: bool = False

@dataclass
class TRLossConfig:
    var_target_init: str = "ones"  # options: "ones", "rand"
    var_sample_factor: float = 1.0
    bidirectional_variance_loss: bool = False

    sim_coeff: float = 10.0
    std_coeff: float = 25.0
    cov_coeff: float = 12.0
    lat_coeff: float = 12.0

    cov_matrix_sparsity: float = 0.0
    consider_last_batch_z: bool = False
    sim_within_chunks: bool = False
    use_cov_directly: bool = False
    detach_previous: bool = False

@dataclass
class EncoderConfig:
    layer_dims: Tuple[Tuple[int, int], ...]
    layer_bias: bool = True
    recurrence_depth: int = 1
    activaton_fn: type[nn.Module] = nn.ReLU

@dataclass
class StoreConfig:
    # depends on batch size too
    # 0.0 = use the batches current statistics
    pre_stats_momentum: float = 0.9
    post_stats_momentum: float = 0
    cov_momentum: float = 0
    last_z_momentum: float = 0.0
    overwrite_at_start: bool = False
    batchless_updates: bool = False 

@dataclass
class DataConfig:
    data_path: str = "./data"
    num_workers: int = 8
    pin_memory: bool = False

    encoder_augment: bool = True
    
    batch_size: int = 64
    chunk_size: int = 16 # size of coherent same-class chunks
    coherence: float = 1.0

@dataclass
class Config:
    project_name: str = "experiments_mnist"
    run_name: str = "v12"
    # wandb or csv
    logger: str = "wandb"
    track_representations: bool = False

    seed: int = 42
    # pass -> a simple forward pass
    # sequence -> sequential setup
    problem_type: str = "pass"
    # classification or regression
    head_task: str = "classification"
    head_out_dim: int = 10

    # different encoders are trained in sequence
    # within encoders, layers can be trained concurrently
    train_encoder_concurrently: bool = True
    epochs: int = 5
    head_epochs: int = 10
    lr: float = 1e-3

    data_config: DataConfig = DataConfig()
    trloss_config: TRLossConfig = TRLossConfig()
    batchnorm_config: BatchNormConfig|None = BatchNormConfig()
    store_config: StoreConfig = StoreConfig()

    encoders = [
        EncoderConfig(((28*28, 512), (512, 256)))
    ]
