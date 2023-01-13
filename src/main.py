#!/usr/bin/python
import sys
folder_name = sys.argv[1]

print('early stopping commented out')

from hyperparams import get_hyperparameter_configuration
from load_data import create_data_split
from initialise import initialise_model, setup_tensorboard_and_checkpoints
from train import optimise_model

cfg = get_hyperparameter_configuration()

train_dataset, validate_dataset = create_data_split(cfg)

model, init_params, cfg, key = initialise_model(cfg, train_dataset)

writer, ckpt_dir = setup_tensorboard_and_checkpoints(folder_name)

# optionally reload previously saved train state
if False:

	import reload_checkpoint
	state = reload_state(model, init_params, cfg, ckpt_dir)
	params = state['train_state'].params

# 'cfg': dict(cfg) save pickle maybe

# from jax.config import config
# config.update("jax_debug_nans", False)
# config.update("jax_disable_jit", True)
# type help at a breakpoint() to see available commands
# use xeus-python kernel -- Python 3.9 (XPython) -- for debugging

# import jax
# jax.profiler.start_trace('runs/' + folder_name)
state = optimise_model(model, init_params, train_dataset, validate_dataset, cfg, key, ckpt_dir, writer)
# jax.profiler.stop_trace()

# from jax import numpy as np
# import tensorflow_datasets as tfds
# data = np.array(list(tfds.as_numpy(train_dataset))[0]['image']).reshape(cfg.batch_size,105,105,1)
# output = forward_pass_model(model, params, data, cfg, key)
from utils import forward_pass_model
params = init_params
output = forward_pass_model(model, params, train_dataset, cfg, key)