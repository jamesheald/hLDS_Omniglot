from jax import random
import jax.numpy as np
from flax import linen as nn

def keyGen(key, n_subkeys):
    
    keys = random.split(key, n_subkeys + 1)
    
    return keys[0], (k for k in keys[1:])

def stabilise_variance(log_var, var_min = 1e-16):
    """
    var_min is added to the variances for numerical stability
    """
    return np.log(np.exp(log_var) + var_min)

def smooth_maximum(p_xy_t, smooth_max_parameter = 1e3):

    p_xy = np.sum(p_xy_t * nn.activation.softmax(p_xy_t * smooth_max_parameter, axis = 0), axis = 0)

    return p_xy