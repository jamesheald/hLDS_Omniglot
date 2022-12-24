from flax import linen as nn
import jax.numpy as np
from jax.tree_util import tree_map
from jax import random, lax
from utils import stabilise_variance

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

        return nn.activation.softmax(z1 / params['t'][0], axis = 1), np.squeeze(z2)

class decoder(nn.Module):
    x_dim: list
    image_dim: list
    T: int
    dt: float

    def setup(self):

        def initialise_LDS_states():
              
            # initialise states to zero
            # the initial state of the top layer will be inferred later by the encoder and so is not set here
            n_layers = len(self.x_dim)
            x0 = []
            for layer in range(1, n_layers):
                
                x0.append(np.zeros(self.x_dim[layer]))

            return x0

        # initialise the states (not learned) of the LDS in the decoder
        self.x0 = initialise_LDS_states()

        # set the x- and y- locations of the gaussian kernels that tile the image space in a grid
        self.grid_x_locations = np.linspace(0.5, self.image_dim[1] - 0.5, self.image_dim[1])
        self.grid_y_locations = np.linspace(0.5, self.image_dim[0] - 0.5, self.image_dim[0])

        # transform list of image dimensions to an array
        self.image_dimensions = np.array(self.image_dim)

    def __call__(self, params, A, z1, z2):

        def decode_one_step(carry, inputs):
            
            def compute_alphas(W, x, b, t):

                return nn.activation.softmax( (W @ x + b) / t, axis = 0)

            def compute_inputs(W, x):

                return W @ x

            def update_state(A, x, alphas, u):

                return x + (np.sum(alphas[:, None, None] * A, axis = 0) @ x + u) * self.dt

            def compute_pen_actions(W, x, b):

                return W @ x + b
            
            def per_pixel_bernoulli_parameter(params, pen_xy, pen_down_log_p):
    
                def log_Gaussian_kernel(x, mu, log_var):
                    """
                    calculate the log likelihood of x under a diagonal Gaussian distribution
                    """
                    log_var = stabilise_variance(log_var)

                    return -0.5 * (x - mu)**2 / np.exp(log_var)

                ll_p_x = log_Gaussian_kernel(pen_xy[0], self.grid_x_locations, params['pen_log_var'])
                ll_p_y = log_Gaussian_kernel(pen_xy[1], self.grid_y_locations, params['pen_log_var'])

                p_xy_t = np.exp(ll_p_x[None,:] + ll_p_y[:,None] + pen_down_log_p)

                return p_xy_t
            
            def update_pen_position(pen_xy, d_xy):
    
                # candidate new pen position
                pen_xy = pen_xy + d_xy

                # align pen position relative to centre of canvas
                pen_xy = pen_xy - self.image_dimensions / 2

                # transform canvas boundaries to -/+ 5
                pen_xy = pen_xy * 2 / self.image_dimensions * 5

                # squash pen position to be within canvas boundaries
                pen_xy = nn.sigmoid(pen_xy)

                # transform canvas boundaries back to their original values
                pen_xy_new = pen_xy * self.image_dimensions

                return pen_xy_new

            x, pen_xy = carry
            top_layer_alphas = inputs

            # compute the alphas
            alphas = tree_map(compute_alphas, params['W_a'], x[:2], params['b_a'], params['t'][1:])

            # prepend the top-layer alphas
            alphas.insert(0, np.squeeze(top_layer_alphas))

            # compute the additive inputs
            u = tree_map(compute_inputs, params['W_u'], x[:2])

            # prepend the top-layer additive inputs
            u.insert(0, np.zeros(x[0].shape))

            # update the states
            x_new = tree_map(update_state, A, x, alphas, u)

            # linear readout from the state at the bottom layer
            pen_actions = compute_pen_actions(params['W_p'], x_new[-1], params['b_p'])

            # pen velocities in x and y directions
            d_xy = pen_actions[:2]

            # log probability that the pen is down (drawing is taking place)
            pen_down_log_p = nn.log_sigmoid(pen_actions[2])

            # calculate the per-pixel bernoulli parameter
            p_xy = per_pixel_bernoulli_parameter(params, pen_xy, pen_down_log_p)

            # update the pen position based on the pen velocity
            pen_xy_new = update_pen_position(pen_xy, d_xy)

            carry = x_new, pen_xy_new
            outputs = alphas, x_new, pen_xy_new, p_xy, pen_down_log_p

            return carry, outputs

        x0 = list(self.x0)

        # prepend the inferred initial state of the top layer to the list of initial states
        x0.insert(0, z2[:])

        pen_xy0 = self.image_dimensions / 2 # initialise pen in centre of canvas

        carry = x0, pen_xy0
        inputs = np.repeat(z1[None,:], self.T, axis = 0)

        _, (alphas, x, pen_xy, p_xy_t, pen_down_log_p) = lax.scan(decode_one_step, carry, inputs)
    
        return {'alphas': alphas,
                'x0': x0,
                'x': x,
                'pen_xy0': pen_xy0,
                'pen_xy': pen_xy,
                'p_xy_t': p_xy_t,
                'pen_down_log_p': pen_down_log_p}

class VAE(nn.Module):
    n_loops_top_layer: int
    x_dim: list
    image_dim: list
    T: int
    dt: float

    def setup(self):
        
        self.encoder = encoder(self.n_loops_top_layer, self.x_dim[0])
        self.sampler = sampler(self.n_loops_top_layer)
        self.decoder = decoder(self.x_dim, self.image_dim, self.T, self.dt)

    def __call__(self, data, params, A, key):

        output_encoder = self.encoder(data[None,:,:,None])
        z1, z2 = self.sampler(output_encoder, params, key)
        output_decoder = self.decoder(params, A, z1, z2)
        
        return output_encoder | output_decoder
