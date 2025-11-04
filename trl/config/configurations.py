from functools import wraps
from copy import deepcopy

from trl.config.config import Config

def change_configuration(fn):
    @wraps
    def wrapper(conf: Config, **kwargs):
        conf = deepcopy(conf)
        fn(conf)
        return conf
    
    return wrapper

@change_configuration
def long_training(conf: Config):
    conf.batch_size = 30
    conf.chunk_size = 6
    conf.lr = 5e-4

@change_configuration
def batchless(conf: Config):
    conf.batch_size = 1
    conf.chunk_size = 3
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

