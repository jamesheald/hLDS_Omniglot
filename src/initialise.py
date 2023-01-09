from jax import random
import jax.numpy as np
from utils import keyGen, construct_dynamics_matrix
from hLDS import VAE
from flax.core.frozen_dict import freeze, unfreeze
from flax.metrics import tensorboard

def initialise_decoder_parameters(cfg, key):
    
    P = []
    S_U = []
    S_V = []
    L = []
    W_u = []
    W_a = []
    b_a = []
    t = []
    
    n_layers = len(cfg.x_dim)
    for layer in range(n_layers):

        key, subkeys = keyGen(key, n_subkeys = 7)
        
        n_loops = cfg.n_loops[layer]
        x_dim = cfg.x_dim[layer]
        
        # parameters of layer-specific P
        p = random.normal(next(subkeys), (x_dim, x_dim))
        
        # set trace of P @ P.T to x_dim
        P.append(p * np.sqrt(x_dim / np.trace(p @ p.T)))
        
        # parameters of layer- and loop-specific S
        u = random.normal(next(subkeys), (n_loops, x_dim, int(x_dim / n_loops)))
        v = random.normal(next(subkeys), (n_loops, x_dim, int(x_dim / n_loops)))
        
        # set variance of elements of S to 1/(n_loops * x_dim)
        s = u @ np.transpose(v, (0, 2, 1)) - v @ np.transpose(u, (0, 2, 1))
        f = 1 / np.std(s, axis = (1, 2))[:, None, None] / np.sqrt(n_loops * x_dim)
        # f = 1 / np.linalg.norm(s, axis = (1, 2))[:, None, None] / n_loops # frobenius norm of each loop 1/n_loops
        S_U.append(u * np.sqrt(f))
        S_V.append(v * np.sqrt(f))

        # parameters of layer- and loop-specific L
        Q, _ = np.linalg.qr(random.normal(next(subkeys), (x_dim, x_dim)))
        L_i = np.split(Q * np.sqrt(n_loops), n_loops, axis = 1)
        L.append(np.stack(L_i, axis = 0))

        # parameters of the mapping from hidden states to alphas, alphas = softmax(W @ x + b, temperature)
        if layer != 0:
            
            # weights for additive inputs
            std_W = 1 / np.sqrt(cfg.x_dim[layer - 1])
            W_u.append(random.normal(next(subkeys), (cfg.x_dim[layer], cfg.x_dim[layer - 1])) * std_W)
            
            # weights for modulatory factors
            W_a.append(random.normal(next(subkeys), (n_loops, cfg.x_dim[layer - 1])) * std_W)

            # bias for modulatory factors
            b_a.append(np.zeros((n_loops)))
            
        if layer == n_layers - 1:
            
            # weights for pen actions
            std_W = 1 / np.sqrt(cfg.x_dim[layer])
            W_p = random.normal(next(subkeys), (3, cfg.x_dim[layer])) * std_W
            
            # bias for pen actions
            b_p = np.zeros((3))
            
        # temperature of layer-specific softmax function
        t.append(1.0)

    return {'P': P, 
            'S_U': S_U,
            'S_V': S_V, 
            'L': L, 
            'W_u': W_u,
            'W_a': W_a, 
            'b_a': b_a,
            't': t,
            'W_p': W_p,
            'b_p': b_p,
            'pen_log_var': cfg.init_pen_log_var}

def initialise_model(cfg, train_dataset):

    # explicitly generate a PRNG key
    key = random.PRNGKey(cfg.jax_seed)

    # generate the required number of subkeys
    key, subkeys = keyGen(key, n_subkeys = 2) 

    cfg.n_batches = len(train_dataset)
    cfg.n_loops = [int(np.ceil(i * cfg.alpha_fraction)) for i in cfg.x_dim]

    # define the model
    model = VAE(n_loops_top_layer = cfg.n_loops[0], x_dim = cfg.x_dim, image_dim = cfg.image_dim, T = cfg.time_steps, dt = cfg.dt)
    
    # initialise the model parameters
    params = {'prior_z_log_var': cfg.prior_z_log_var,
              'decoder': initialise_decoder_parameters(cfg, next(subkeys))}
    init_params = model.init(data = np.ones((1, cfg.image_dim[0], cfg.image_dim[1], 1)), params = params['decoder'], A = construct_dynamics_matrix(params['decoder']),
                             key = next(subkeys), rngs = {'params': random.PRNGKey(0)})['params']

    # concatenate all parameters into a single dictionary
    init_params = unfreeze(init_params)
    init_params = init_params | params
    init_params = freeze(init_params)

    return model, init_params, cfg, key

def setup_tensorboard_and_checkpoints(folder_name):

    writer = tensorboard.SummaryWriter('runs/' + folder_name) # to view tensorboard results, call 'tensorboard --logdir=.' in runs folder (not in python)

    ckpt_dir = 'tmp/flax-checkpointing/' + folder_name

    return writer, ckpt_dir