from dataclasses import dataclass, field
from typing import Tuple
from torch import nn as nn
import torch
import torch.nn.functional as F

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

    # good if detach_previous is False
    # sim_coeff: float = 10.0
    # std_coeff: float = 25.0
    # cov_coeff: float = 12.0
    # lat_coeff: float = 12.0

    # if detach_previous is True
    sim_coeff: float = 1.0
    std_coeff: float = 2.5
    cov_coeff: float = 1.0
    lat_coeff: float = 1.2

    cov_matrix_sparsity: float = 0.0
    consider_last_batch_z: bool = False
    # Keep naming as requested in current experiments.
    use_chunk_paritions: bool = False
    use_cov_directly: bool = False
    detach_previous: bool = True
    use_trace_activation: bool = False
    trace_decay: float = 0.9
    lateral_shift: bool = False
    lateral_shift_cov_target: bool = False
    decorrelation_eps: float = 1e-6
    # Sequence-only: if True, average std-loss contribution across time before batch reduction.
    # Keeps std-loss scale consistent with sequence length.
    sequence_std_mean_over_time: bool = False

    variance_hinge_fn = lambda self, x: F.relu(x)
    sim_loss_fn = lambda self, a, b: (a-b)**2

@dataclass
class EncoderConfig:
    layer_dims: Tuple[Tuple[int, int], ...]
    layer_bias: bool = True
    recurrence_depth: int = 1
    activation_fn: type[nn.Module] = nn.ReLU
    activation_fn_requires_dim: bool = False

@dataclass
class StoreConfig:
    # depends on batch size too
    # 0.0 = use the batches current statistics
    pre_stats_momentum: float = 0.9
    post_stats_momentum: float = 0
    cov_momentum: float = 0
    last_z_momentum: float = 0.0
    trace_momentum: float = 0.9
    overwrite_at_start: bool = False
    batchless_updates: bool = False 

    device: str = "cpu"

@dataclass
class DataConfig:
    data_path: str = "./data"
    dataset_name: str = "mnist"
    # If True, try to download torchvision datasets when missing.
    # If False, require datasets to already exist under data_path.
    allow_download: bool = True
    num_workers: int = 1
    pin_memory: bool = False

    encoder_augment: bool = False
    
    batch_size: int = 64
    chunk_size: int = 16 # size of coherent same-class chunks
    coherence: float = 1.0
    # If True, use class-coherent chunk ordering via CoherentSampler.
    # If False, use standard shuffled minibatches.
    use_coherent_sampler: bool = True
    # Head training uses plain shuffled batches by default.
    use_coherent_sampler_for_head: bool = False

    # Generic temporal-order toggle for naturally sequential datasets (e.g. PAMAP2):
    # True keeps chronological sample ordering, False shuffles training samples.
    temporal_coherence_ordering: bool = False
    # PAMAP2-specific downsampling stride (100 Hz -> 10 Hz when stride=10).
    pamap2_stride: int = 10
    # If True, split PAMAP2 by subject IDs (train/val/test subject-disjoint).
    pamap2_subject_disjoint_split: bool = True
    pamap2_split_seed: int = 42

@dataclass
class Config:
    project_name: str = "experiments_mnist"
    run_name: str = "v12"
    # wandb or csv
    logger: str = "wandb"
    track_representations: bool = False
    # specify which layers the head uses
    # False: uses last layer, True: use all layers
    # finish_setup converts it to either the specified layers or None
    head_use_layers: list|bool|None = False
    encoder_optim: type[torch.optim.Optimizer] = torch.optim.SGD
    head_optim: type[torch.optim.Optimizer] = torch.optim.SGD

    seed: int = 42
    # pass -> a simple forward pass
    # sequence -> sequential setup
    problem_type: str = "pass"
    # sequence encoder mode:
    # True -> recurrent TRSeqEncoder (legacy),
    # False -> elementwise per-timestep encoder (proposal setup).
    sequence_recurrent_encoder: bool = True
    # classification or regression
    head_task: str = "classification"
    head_out_dim: int = 10
    # Backprop sequence baseline: if False use MLP-style recurrent mapping, if True use linear recurrent mapping.
    bp_sequence_linear: bool = False
    # Sequence head temporal fusion: "none", "ema", "residual_gate", "concat_linear", or "concat_mlp".
    temporal_fusion_mode: str = "none"
    temporal_fusion_alpha: float = 0.5
    temporal_fusion_hidden_dim: int = 0
    temporal_fusion_trl_coeff: float = 1.0

    # different encoders are trained in sequence
    # within encoders, layers can be trained concurrently
    train_encoder_concurrently: bool = True
    epochs: int = 10
    head_epochs: int = 10
    lr: float = 3e-4
    head_lr: float = 0.1
    encoder_grad_clip_norm: float = 1.0
    encoder_lat_lr_factor: float = 0.03

    data_config: DataConfig = field(default_factory=DataConfig)
    trloss_config: TRLossConfig = field(default_factory=TRLossConfig)
    batchnorm_config: BatchNormConfig|None = None
    store_config: StoreConfig = field(default_factory=StoreConfig)

    encoders = [
        EncoderConfig(((28*28, 512), (512, 256)))
    ]

    def setup_head_use_layers(self):
        if self.head_use_layers is True:
            self.head_use_layers = [i for i in range(len(self.encoders[-1].layer_dims))]
        elif self.head_use_layers is False:
            self.head_use_layers = None

