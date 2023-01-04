#!/usr/bin/python
import sys
folder_name = sys.argv[1]

from hyperparams import get_hyperparameter_configuration
from load_data import create_data_split
from initialise import initialise_model, setup_tensorboard_and_checkpoints
from train import optimise_model

cfg = get_hyperparameter_configuration()

train_dataset, validate_dataset, test_dataset, _ = create_data_split(cfg)

model, init_params, cfg, key = initialise_model(cfg, train_dataset)

writer, ckpt_dir = setup_tensorboard_and_checkpoints(folder_name)

# from jax.config import config
# config.update("jax_debug_nans", False)
# config.update("jax_disable_jit", False)
# type help at a breakpoint() to see available commands
# use xeus-python kernel -- Python 3.9 (XPython) -- for debugging

# import jax
# jax.profiler.start_trace('runs/' + folder_name)
state, losses = optimise_model(model, init_params, train_dataset, validate_dataset, cfg, key, ckpt_dir, writer)
# jax.profiler.stop_trace()

# # restore checkpoint
# from flax.training import checkpoints, train_state
# ckpt = {'train_state': state, 'losses': losses, 'cfg': dict(cfg)}
# restored_state = checkpoints.restore_checkpoint(ckpt_dir = ckpt_dir, target = ckpt)