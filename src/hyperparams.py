from ml_collections import config_dict
import jax.numpy as np

def get_hyperparameter_configuration():
  
    cfg = config_dict.ConfigDict()

    # model
    cfg.x_dim = [20, 50, 200]
    cfg.alpha_fraction = 0.1
    cfg.dt = 0.01
    cfg.time_steps = 100
    cfg.init_pen_log_var = 10.0
    cfg.image_dim = [105, 105]
    cfg.prior_z_log_var = -2.0
    cfg.jax_seed = 0

    # data
    cfg.percent_data_to_use = 1
    cfg.fraction_for_validation = 0.2
    cfg.batch_size = 32
    cfg.tfds_seed = 0

    # images to write to tensorboard
    cfg.square_image_grid_size = 7

    # optimisation
    cfg.kl_warmup_start = 500
    cfg.kl_warmup_end = 1000
    cfg.kl_min = 0.01
    cfg.kl_max = 1
    cfg.adam_b1 = 0.9
    cfg.adam_b2 = 0.999                        
    cfg.adam_eps = 1e-8
    cfg.weight_decay = 0.0001
    cfg.max_grad_norm = 10
    cfg.step_size = 0.001
    cfg.decay_steps = 1
    cfg.decay_factor = 0.9999
    cfg.print_every = 1
    cfg.n_epochs = 2
    cfg.min_delta = 1e-3
    cfg.patience = 20

    return cfg