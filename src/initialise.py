from jax import random
import jax.numpy as np
from utils import keyGen, construct_dynamics_matrix
from hLDS import VAE
from flax.core.frozen_dict import freeze, unfreeze

def initialise_decoder_parameters(args, key):
    
    P = []
    S_U = []
    S_V = []
    L = []
    W_u = []
    W_a = []
    b_a = []
    gamma = []
    
    n_layers = len(args.x_dim)
    for layer in range(n_layers):

        key, subkeys = keyGen(key, n_subkeys = 7)
        
        n_loops = args.n_loops[layer]
        x_dim = args.x_dim[layer]
        
        # parameters of layer-specific P
        p = random.normal(next(subkeys), (x_dim, x_dim))
        
        # set trace of P @ P.T to x_dim
        P.append(p * np.sqrt(x_dim / np.trace(p @ p.T)))
        
        # parameters of layer- and loop-specific S
        u = random.normal(next(subkeys), (n_loops, x_dim, int(x_dim / n_loops)))
        v = random.normal(next(subkeys), (n_loops, x_dim, int(x_dim / n_loops)))
        
        s = u @ np.transpose(v, (0, 2, 1)) - v @ np.transpose(u, (0, 2, 1))
        # f = 1 / np.std(s, axis = (1, 2))[:, None, None] / np.sqrt(n_loops * x_dim) # set variance of elements of each loop of S to 1/(n_loops * x_dim)
        f = 1 / np.linalg.norm(s, axis = (1, 2))[:, None, None] / np.sqrt(n_loops) # set frobenius norm squared of each loop to 1/n_loops
        S_U.append(u * np.sqrt(f))
        S_V.append(v * np.sqrt(f))

        # parameters of layer- and loop-specific L
        Q, _ = np.linalg.qr(random.normal(next(subkeys), (x_dim, x_dim)))
        L_i = np.split(Q * np.sqrt(n_loops), n_loops, axis = 1)
        L.append(np.stack(L_i, axis = 0))

        # parameters of the mapping from hidden states to alphas, alphas = softmax(W @ x + b, temperature)
        if layer != 0:
            
            # weights for additive inputs
            std_W = 1 / np.sqrt(args.x_dim[layer - 1])
            W_u.append(random.normal(next(subkeys), (args.x_dim[layer], args.x_dim[layer - 1])) * std_W)
            
            # weights for modulatory factors
            W_a.append(random.normal(next(subkeys), (n_loops, args.x_dim[layer - 1])) * std_W)

            # bias for modulatory factors
            b_a.append(np.zeros((n_loops)))
            
        if layer == n_layers - 1:
            
            # weights for pen actions
            std_W = 1 / np.sqrt(args.x_dim[layer])
            W_p = random.normal(next(subkeys), (3, args.x_dim[layer])) * std_W
            
            # bias for pen actions
            b_p = np.zeros((3))
            
        # coefficient for interpolating between loop dynamics and the dynamics of a negative identity matrix 
        # this shifts the real part of the eigenvalues of the dynamics matrix to the left (for numerical stability)
        gamma.append(0.0)

    return {'P': P, 
            'S_U': S_U,
            'S_V': S_V, 
            'L': L, 
            'W_u': W_u,
            'W_a': W_a, 
            'b_a': b_a,
            'W_p': W_p,
            'b_p': b_p,
            'gamma': gamma,
            'pen_log_var': args.init_pen_log_var}

def initialise_model(args, train_dataset):

    # explicitly generate a PRNG key
    key = random.PRNGKey(args.jax_seed)

    # generate the required number of subkeys
    key, subkeys = keyGen(key, n_subkeys = 2) 

    # args.n_batches = len(train_dataset) # TFDS change
    args.n_batches = 1
    args.n_loops = [int(np.ceil(i * args.alpha_fraction)) for i in args.x_dim]

    # define the model
    model = VAE(n_loops_top_layer = args.n_loops[0], x_dim = args.x_dim, image_dim = args.image_dim, T = args.time_steps, dt = args.dt, tau = args.tau)
    
    # initialise the model parameters
    params = {'prior_z_log_var': args.prior_z_log_var,
              'decoder': initialise_decoder_parameters(args, next(subkeys))}
    A, gamma = construct_dynamics_matrix(params['decoder'])
    init_params = model.init(data = np.ones((args.image_dim[0], args.image_dim[1], 2)), params = params['decoder'], A = A,
                             gamma = gamma, key = next(subkeys), rngs = {'params': random.PRNGKey(0)})['params']

    # concatenate all parameters into a single dictionary
    init_params = unfreeze(init_params)
    init_params = init_params | params
    init_params = freeze(init_params)

    return model, init_params, args, key