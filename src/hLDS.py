from flax import linen as nn
import jax.numpy as np
from jax.tree_util import tree_map
from jax import random, lax
from utils import stabilise_variance, bound_variable

class encoder(nn.Module):
    n_loops_top_layer: int
    x_dim_top_layer: int

    @nn.compact
    def __call__(self, x):
        
        # CNN architecture based on 'End-to-End Training of Deep Visuomotor Policies' paper
        x = nn.Conv(features = 64, kernel_size = (7, 7))(x)
        x = nn.relu(x)
        x = nn.Conv(features = 32, kernel_size = (5, 5))(x)
        x = nn.relu(x)
        x = nn.Conv(features = 32, kernel_size = (5, 5))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1)) # flatten
        x = nn.Dense(features = 64)(x)
        x = nn.relu(x)
        x = nn.Dense(features = 40)(x)
        x = nn.relu(x)
        x = nn.Dense(features = 40)(x)
        x = nn.relu(x)
        x = nn.Dense(features = (self.n_loops_top_layer + self.x_dim_top_layer) * 2)(x)
        
        # mean and log variances of Gaussian distribution over latents
        z_mean, z_log_var = np.split(x, 2, axis = 1)
        
        return {'z_mean': z_mean, 'z_log_var': z_log_var}
    
class sampler(nn.Module):
    n_loops_top_layer: int
    
    @nn.compact
    def __call__(self, output, params, key):
        
        def sample_diag_Gaussian(mean, log_var, key):
            """
            sample from a diagonal Gaussian distribution
            """
            log_var = stabilise_variance(log_var)

            return mean + np.exp(0.5 * log_var) * random.normal(key, mean.shape)

        # sample the latents
        z = sample_diag_Gaussian(output['z_mean'], output['z_log_var'], key)

        # split the latents into top-layer alphas, softmax(z1), and initial state, z2
        z1, z2 = np.split(z, [self.n_loops_top_layer], axis = 1)

        return nn.activation.softmax(z1, axis = 1), np.squeeze(z2)

class myosuite_dynamics(nn.Module):
    carry_dim: int
    fingertip_centre: list
    fingertip_range: list

    def setup(self):

        # model 
        self.Dense = nn.Dense(3)
        self.GRUCell = nn.GRUCell()
        self.param('carry_init', lambda rng, shape: np.zeros(shape), (self.carry_dim,))

        # transform fingertip_centre and fingertip_range lists to arrays
        self.tip_centre = np.array(self.fingertip_centre)
        self.tip_range = np.array(self.fingertip_range)

    def __call__(self, carry, inputs):
        
        # # set the initial carry to be a learnable parameter
        # if carry is None:

        #     # carry = self.param('carry_init', np.zeros((self.carry_dim,)), (self.carry_dim,))
        #     # carry = self.param('carry_init', initializers.zeros, (self.carry_dim,))
        #     carry = self.param('carry_init', lambda rng, shape: np.zeros(shape), (self.carry_dim,))

        # # apply dropout to inputs
        # # inputs = nn.Dropout(rate = self.dropout_rate)(inputs, deterministic = deterministic)

        # # update the GRU
        # carry, outputs = nn.GRUCell()(carry, inputs)

        # # readout the fingertip position from the GRU
        # fingertip_xyz = nn.Dense(3)(outputs)

        # # bound the fingertip position to be between x_centre - x_range / 2 and x_centre + x_range / 2
        # fingertip_xyz = bound_variable(x = fingertip_xyz, x_centre = self.tip_centre, x_range = self.tip_range)

        # apply dropout to inputs
        # inputs = nn.Dropout(rate = self.dropout_rate)(inputs, deterministic = deterministic)

        # update the GRU
        carry, outputs = self.GRUCell(carry, inputs)

        # readout the fingertip position from the GRU
        fingertip_xyz = self.Dense(outputs)

        # bound the fingertip position to be between x_centre - x_range / 2 and x_centre + x_range / 2
        # fingertip_xyz = bound_variable(x = fingertip_xyz, x_centre = self.tip_centre, x_range = self.tip_range)
        
        return carry, fingertip_xyz

class decoder(nn.Module):
    x_dim: list
    image_dim: list
    T: int
    dt: float
    tau: float

    def setup(self):

        # set the x- and y- locations of the gaussian kernels that tile the image space in a grid
        self.grid_x_locations = np.linspace(0.5, self.image_dim[1] - 0.5, self.image_dim[1])
        self.grid_y_locations = np.linspace(0.5, self.image_dim[0] - 0.5, self.image_dim[0])

        # transform list of image dimensions to an array
        self.image_dimensions = np.array(self.image_dim)

    def __call__(self, params, A, gamma, z1, z2, state_myo):

        def decode_one_step(carry, inputs):
            
            def compute_alphas(x, W, b):

                return nn.activation.softmax(W @ x + b, axis = 0)

            def compute_inputs(W, x):

                return W @ x

            def update_state(A, gamma, x, alphas, u):

            	aAx = np.sum(alphas[:,None] * (A @ x), axis = 0)

            	return x + ((1 - gamma) * aAx - gamma * x + u) * self.dt / self.tau

            def compute_muscle_inputs(x, W, b):

                return W @ x + b

            def compute_pen_state(x, W, b):

                pen_state = W @ x + b

                # candidate pen position in the x- and y-dimensions, scaled and shifted based on canvas dimensions
                pen_xy = pen_state[:2] * 20 + np.array([105,105]) / 2

                # squash the pen position to be within the boundaries of the canvas
                pen_xy = bound_variable(x = pen_xy, x_centre = self.image_dimensions / 2, x_range = self.image_dimensions)

                # log probability that the pen is down (drawing is taking place)
                pen_down_log_p = nn.log_sigmoid(pen_state[2])

                return pen_xy, pen_down_log_p
            
            def per_pixel_bernoulli_parameter(params, pen_xy, pen_down_log_p):
    
                def log_Gaussian_kernel(x, mu, log_var):
                    """
                    calculate the log likelihood of x under a diagonal Gaussian distribution
                    """
                    log_var = stabilise_variance(log_var)

                    return -0.5 * (x - mu)**2 / np.exp(log_var)

                def stabilise_cross_entropy(p_xy_t, p_min = 1e-6):
                    """
                    squash the per pixel bernoulli parameter between p_min and 1 - p_min for numerical stability
                    """

                    return p_xy_t * (1 - p_min * 2) + p_min

                # pen position is in image coordinates (distance from top of image, distance from left of image)
                ll_p_y = log_Gaussian_kernel(pen_xy[0], self.grid_y_locations, params['pen_log_var'])
                ll_p_x = log_Gaussian_kernel(pen_xy[1], self.grid_x_locations, params['pen_log_var'])

                p_xy_t = np.exp(ll_p_y[:,None] + ll_p_x[None,:] + pen_down_log_p)

                p_xy_t = stabilise_cross_entropy(p_xy_t)

                return p_xy_t

            x, myosuite_carry = carry
            top_layer_alphas = inputs

            # compute the alphas
            alphas = [np.squeeze(top_layer_alphas), *tree_map(compute_alphas, x[:2], params['W_a'], params['b_a'])]

            # compute the additive inputs
            u = [np.zeros(x[0].shape), *tree_map(compute_inputs, params['W_u'], x[:2])]

            # update the states
            x = tree_map(update_state, A, gamma, x, alphas, u)

            # linear readout from the state at the bottom layer
            muscle_inputs = compute_muscle_inputs(x[-1], params['W_m'], params['b_m'])

            # predict the myoFinger fingertip position using the approximate (learned) myosuite dynamics model
            myosuite_carry, fingertip_xyz = state_myo.apply_fn(state_myo.params, myosuite_carry, muscle_inputs)

            # map the fingertip position to the pen state
            pen_xy, pen_down_log_p = compute_pen_state(fingertip_xyz, params['W_p'], params['b_p'])

            # calculate the per-pixel bernoulli parameter
            p_xy = per_pixel_bernoulli_parameter(params, pen_xy, pen_down_log_p)

            carry = x, myosuite_carry
            outputs = alphas, u, x, muscle_inputs, fingertip_xyz, pen_xy, pen_down_log_p, p_xy

            return carry, outputs

        # initial LDS states
        # state of top layer is inferred by the encoder, states of lower layers are set to 0
        n_layers = len(self.x_dim)
        x0 = [z2[:], *[np.zeros(self.x_dim[layer]) for layer in range(1, n_layers)]]

        carry = x0, state_myo.params['params']['carry_init']
        inputs = np.repeat(z1[None,:], self.T, axis = 0)

        _, (alphas, u, x, muscle_inputs, fingertip_xyz, pen_xy, pen_down_log_p, p_xy_t) = lax.scan(decode_one_step, carry, inputs)
    
        return {'alphas': alphas,
                'u': u,
                'x0': x0,
                'x': x,
                'muscle_inputs': muscle_inputs,
                'fingertip_xyz': fingertip_xyz,
                'pen_xy': pen_xy,
                'pen_down_log_p': pen_down_log_p,
                'p_xy_t': p_xy_t}

class VAE(nn.Module):
    n_loops_top_layer: int
    x_dim: list
    image_dim: list
    T: int
    dt: float
    tau: float

    def setup(self):
        
        self.encoder = encoder(self.n_loops_top_layer, self.x_dim[0])
        self.sampler = sampler(self.n_loops_top_layer)
        self.decoder = decoder(self.x_dim, self.image_dim, self.T, self.dt, self.tau)

    def __call__(self, data, params, A, gamma, state_myo, key):

        output_encoder = self.encoder(data[None,:,:,None])
        z1, z2 = self.sampler(output_encoder, params, key)
        output_decoder = self.decoder(params, A, gamma, z1, z2, state_myo)
        
        return output_encoder | output_decoder
