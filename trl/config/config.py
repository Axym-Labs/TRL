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
class TCLossConfig:
    var_target_init: str = "ones"  # options: "ones", "rand"
    var_sample_factor: float = 1.0
    bidirectional_variance_loss: bool = False

    sim_coeff: float = 10.0
    std_coeff: float = 25.0
    cov_coeff: float = 12.0
    lat_coeff: float = 12.0

    cov_matrix_sparsity: float = 0.0
    consider_last_batch_z: bool = True

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
    post_stats_momentum: float = 0.0
    cov_momentum: float = 0.0
    last_z_momentum: float = 0.0
    overwrite_at_start: bool = False
    batchless_updates: bool = False 

@dataclass
class DataConfig:
    data_path: str = "./data"
    num_workers: int = 4
    pin_memory: bool = True
    
    batch_size: int = 64
    chunk_size: int = 16 # size of coherent same-class chunks
    coherence: float = 1.0

@dataclass
class Config:
    project_name: str = "experiments_mnist"
    run_name: str = "v12_recurrent"
    seed: int = 42

    # different encoders are trained in sequence
    # within encoders, layer can be trained concurrently
    train_encoder_concurrently: bool = True
    epochs: int = 20
    classifier_epochs: int = 20
    lr: float = 1e-3

    data_config: DataConfig = DataConfig()
    tcloss_config: TCLossConfig = TCLossConfig()
    batchnorm_config: BatchNormConfig|None = BatchNormConfig()
    store_config: StoreConfig = StoreConfig()

    encoders = [
        # old version
        EncoderConfig(((28*28, 512), (512, 256))) 
        
        # EncoderConfig(((28*28, 128),), activaton_fn=nn.ReLU), # downsizer
        # EncoderConfig(tuple([(128, 128) for _ in range(9)]), recurrence_depth=1, activaton_fn=nn.ReLU)
    ]
