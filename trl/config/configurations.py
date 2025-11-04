from functools import wraps
from copy import deepcopy

from trl.config.config import Config, EncoderConfig

def change_configuration(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        conf_new = Config()
        fn(conf_new, *args, **kwargs)
        conf_new.run_name = conf_new.run_name + " " + fn.__name__
        return conf_new
    
    return wrapper

@change_configuration
def intermediate_length_run(conf: Config):
    conf.epochs = 20
    conf.head_epochs = 20

@change_configuration
def long_training(conf: Config):
    conf.data_config.batch_size = 30
    conf.data_config.chunk_size = 6
    conf.epochs = 60
    conf.head_epochs = 60
    conf.lr = 5e-4

@change_configuration
def batchless(conf: Config):
    conf.data_config.batch_size = 1
    conf.data_config.chunk_size = 3
    conf.store_config.pre_stats_momentum = 0.9994
    conf.store_config.post_stats_momentum = 0.9994
    conf.store_config.cov_momentum = 0.9994
    conf.store_config.batchless_updates = True
    conf.tcloss_config.consider_last_batch_z = True
    if conf.batchnorm_config is not None:
        conf.batchnorm_config.use_batch_statistics_training = False

@change_configuration
def minimal_batchnorm(conf: Config):
    assert conf.batchnorm_config is not None

    conf.batchnorm_config.bias_parameter = False
    conf.batchnorm_config.scale_parameter = False
    conf.batchnorm_config.use_variance = False
    # mean stays True
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
        EncoderConfig(((28, 128),(128,64)))
    ]
    conf.head_out_dim = 28
    conf.head_task = "regression"
    conf.problem_type = "sequence"
    print("bla")

