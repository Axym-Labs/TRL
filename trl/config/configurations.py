from functools import wraps
import torch

from trl.config.config import Config, EncoderConfig, BatchNormConfig

def change_configuration(fn):
    @wraps(fn)
    def wrapper(conf=None, **kwargs):
        conf_new = conf or Config()
        fn(conf_new, **kwargs)
        if fn.__name__ != "finish_setup":
            conf_new.run_name = conf_new.run_name + " " + fn.__name__
        return conf_new
    
    return wrapper

@change_configuration
def intermediate_length_run(conf: Config):
    conf.epochs = 20
    conf.head_epochs = 20

@change_configuration
def long_training(conf: Config):
    conf.epochs = 60
    conf.head_epochs = 60

@change_configuration
def batchless(conf: Config):
    conf.data_config.batch_size = 1
    conf.data_config.chunk_size = 3
    conf.store_config.pre_stats_momentum = 0.99 # 0.9994
    conf.store_config.post_stats_momentum = 0.99 # 0.9994
    conf.store_config.cov_momentum = 0.99 # 0.9994
    conf.lr /= 50
    conf.trloss_config.std_coeff /= 2
    conf.store_config.batchless_updates = True
    conf.trloss_config.consider_last_batch_z = True
    if conf.batchnorm_config is not None:
        conf.batchnorm_config.use_batch_statistics_training = False

@change_configuration
def minimal_batchnorm(conf: Config):
    assert conf.batchnorm_config is None, "Overwriting existing batchnorm config"
    conf.batchnorm_config = BatchNormConfig()

    conf.batchnorm_config.bias_parameter = False
    conf.batchnorm_config.scale_parameter = False
    conf.batchnorm_config.use_variance = False
    # mean stays True
    conf.batchnorm_config.use_batch_statistics_training = False
    conf.batchnorm_config.detach_batch_statistics = True

@change_configuration
def mnist_deep_net(conf: Config):
        # testing depth
    conf.encoders = [
        EncoderConfig(((28*28, 128),)), # downsizer
        EncoderConfig(tuple([(128, 128) for _ in range(3)]), recurrence_depth=1),
        EncoderConfig(tuple([(128, 128) for _ in range(3)]), recurrence_depth=1),
        EncoderConfig(tuple([(128, 128) for _ in range(3)]), recurrence_depth=1),
    ]

@change_configuration
def mnist_rnn_setup(conf: Config):
    conf.encoders = [
        EncoderConfig(((28+64, 128),(128,64)))
    ]
    conf.data_config.dataset_name = "mnist-rows"
    conf.head_out_dim = 28
    conf.head_task = "regression"
    conf.problem_type = "sequence"
    conf.sequence_recurrent_encoder = True


@change_configuration
def mnist_local_state_setup(conf: Config):
    # Elementwise encoder: x_t -> y_t, no recurrent hidden state inside encoder.
    conf.encoders = [
        EncoderConfig(((28, 128), (128, 64)))
    ]
    conf.data_config.dataset_name = "mnist-rows"
    conf.head_out_dim = 28
    conf.head_task = "regression"
    conf.problem_type = "sequence"
    conf.sequence_recurrent_encoder = False

@change_configuration
def mnist_backprop_comparison_tuning(conf: Config):
    intermediate_length_run(conf)
    conf.logger = "csv"

@change_configuration
def ff_scale_network(conf: Config):
    conf.encoders = [
        EncoderConfig(((28*28, 2000), *[(2000, 2000) for _ in range(3)])),
    ]
    conf.trloss_config.cov_matrix_sparsity = 0.5

@change_configuration
def eqprop_scale_network(conf: Config):
    conf.encoders = [
        EncoderConfig(((28*28, 500), (500, 500), (500, 500)))
    ]

@change_configuration
def standard_setup(conf: Config):
    conf.head_use_layers = True
    conf.encoder_optim = torch.optim.SGD
    conf.train_encoder_concurrently = False
    conf.trloss_config.use_chunk_paritions = True
    # use_cov_directly is not beneficial
    # counteracts doulble-z-term in MSE which favors collapse
    conf.trloss_config.detach_previous = False
    conf.trloss_config.std_coeff *= 2 # because detach_previous=False

@change_configuration
def old_setup(conf: Config):
    conf.head_use_layers = True
    conf.encoder_optim = torch.optim.Adam

@change_configuration
def last_layer_head(conf: Config):
    conf.head_use_layers = False

@change_configuration
def enable_trace(conf: Config, decay: float = 0.9):
    conf.trloss_config.use_trace_activation = True
    conf.trloss_config.trace_decay = decay
    conf.store_config.trace_momentum = decay

@change_configuration
def enable_lateral_shift(conf: Config):
    conf.trloss_config.lateral_shift = True

@change_configuration
def enable_lateral_shift_cov_target(conf: Config):
    conf.trloss_config.lateral_shift_cov_target = True

@change_configuration
def aug_and_rbn_setup(conf: Config):
    conf.data_config.encoder_augment = True
    conf.batchnorm_config = BatchNormConfig()

    conf.trloss_config.std_coeff /= 2 # because of batchnorm
    conf.lr = 4e-4

@change_configuration
def sgd_optim(conf: Config):
    conf.encoder_optim = torch.optim.SGD
    conf.trloss_config.use_cov_directly = True
    conf.encoder_optim = torch.optim.SGD
    conf.trloss_config.std_coeff *= 2

@change_configuration
def temporal_coherence_ordering(conf: Config, enabled: bool = True):
    conf.data_config.temporal_coherence_ordering = enabled

@change_configuration
def pamap2_setup(conf: Config):
    # PAMAP2 protocol files: per-sample activity classification on wearable streams.
    conf.data_config.dataset_name = "pamap2"
    conf.problem_type = "pass"
    conf.head_task = "classification"
    conf.head_out_dim = 12
    conf.encoders = [
        EncoderConfig(((52, 256), (256, 128)))
    ]
    conf.data_config.batch_size = 512
    conf.data_config.use_coherent_sampler = False
    conf.data_config.use_coherent_sampler_for_head = False
    conf.data_config.temporal_coherence_ordering = True
    conf.lr = 1e-3
    conf.head_lr = 1e-2

@change_configuration
def finish_setup(conf: Config):
    conf.setup_head_use_layers()
