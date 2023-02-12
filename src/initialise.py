from jax import random
import jax.numpy as np
from utils import keyGen, construct_dynamics_matrix
from hLDS import VAE, myosuite_dynamics
from flax.core.frozen_dict import freeze, unfreeze
from train import get_train_state

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

        key, subkeys = keyGen(key, n_subkeys = 8)
        
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
            
            # weights for muscle inputs
            std_W = 1 / np.sqrt(args.x_dim[layer])
            W_m = random.normal(next(subkeys), (5, args.x_dim[layer])) * std_W
            
            # bias for muscle inputs
            b_m = np.zeros((5))

            # weights for pen state
            std_W = 1 / np.sqrt(3)
            W_p = random.normal(next(subkeys), (3, 3)) * std_W
            
            # bias for pen state
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
            'W_m': W_m,
            'b_m': b_m,
            'W_p': W_p,
            'b_p': b_p,
            'gamma': gamma,
            'pen_log_var': args.init_pen_log_var}

def initialise_model(args, train_dataset):

    # explicitly generate a PRNG key
    key = random.PRNGKey(args.jax_seed)

    # generate the required number of subkeys
    key, subkeys = keyGen(key, n_subkeys = 2) 

    # define some additional hyperparameters
    args.n_batches = len(train_dataset)
    args.n_loops = [int(np.ceil(i * args.alpha_fraction)) for i in args.x_dim]

    # define the myosuite dynamics model and initialise its parameters
    model_myo = myosuite_dynamics(carry_dim = args.carry_dim, fingertip_centre = args.fingertip_centre, fingertip_range = args.fingertip_range)
    params_myo = model_myo.init(carry = np.ones((args.carry_dim,)), inputs = np.ones((5,)), rngs = {'params': random.PRNGKey(0)})
    
    # define the VAE model and initialise its parameters
    model_vae = VAE(n_loops_top_layer = args.n_loops[0], x_dim = args.x_dim, image_dim = args.image_dim, T = args.time_steps, dt = args.LDS_dt, tau = args.tau)
    params = {'prior_z_log_var': args.prior_z_log_var,
              'decoder': initialise_decoder_parameters(args, next(subkeys))}
    A, gamma = construct_dynamics_matrix(params['decoder'])
    state_myo, _ = get_train_state(([], model_myo), ([], params_myo), args, 'myosuite')
    params_vae = model_vae.init(data = np.ones((args.image_dim[0], args.image_dim[1])), params = params['decoder'],
                                A = A, gamma = gamma, state_myo = state_myo, key = next(subkeys), rngs = {'params': random.PRNGKey(0)})

    # concatenate all of the VAE parameters into a single dictionary
    params_vae = unfreeze(params_vae)
    params_vae = params_vae | params
    params_vae = freeze(params_vae)

    return (model_vae, model_myo), (params_vae, params_myo), state_myo, args, key
