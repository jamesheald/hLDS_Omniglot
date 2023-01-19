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

        return nn.activation.softmax(z1, axis = 1), np.squeeze(z2)

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

    def __call__(self, data, params, A, gamma, z1, z2):

        def decode_one_step(carry, inputs):
            
            def compute_alphas(W, x, b):

                return nn.activation.softmax(W @ x + b, axis = 0)

            def compute_inputs(W, x):

                return W @ x

            def update_state(A, gamma, x, alphas, u):

            	aAx = np.sum(alphas[:,None] * (A @ x), axis = 0)

            	return x + ((1 - gamma) * aAx - gamma * x + u) * self.dt / self.tau

            def compute_pen_actions(W, x, b):

                return W @ x + b
            
            def per_pixel_bernoulli_parameter(params, pen_xy, pen_down_log_p):
    
                def log_Gaussian_kernel(x, mu, log_var):
                    """
                    calculate the log likelihood of x under a diagonal Gaussian distribution
                    """
                    log_var = stabilise_variance(log_var)

                    return -0.5 * (x - mu)**2 / np.exp(log_var)

                def stabilise_cross_entropy(p_xy_t, p_min = 1e-16):
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
            
            def update_pen_position(pen_xy, d_xy):
    
                # candidate new pen position
                pen_xy = pen_xy + d_xy

                # align pen position relative to centre of canvas
                pen_xy = pen_xy - self.image_dimensions / 2

                # transform canvas boundaries to -/+ f (this determines degree of squashing)
                f = 2
                pen_xy = pen_xy * 2 / self.image_dimensions * f

                # squash pen position to be within canvas boundaries
                pen_xy = nn.sigmoid(pen_xy)

                # transform canvas boundaries back to their original values
                pen_xy_new = pen_xy * self.image_dimensions

                # x = np.linspace(0,105,1000)
                # a = np.array([105])
                # y = x - a / 2
                # y = y * 2 / a * 2
                # y = 1/(1 + np.exp(-y))
                # y = y * a
                # plt.plot(x,x)
                # plt.plot(x,y)
                # plt.show()

                return pen_xy_new

            x, pen_xy = carry
            top_layer_alphas = inputs

            # compute the alphas
            alphas = [np.squeeze(top_layer_alphas), *tree_map(compute_alphas, params['W_a'], x[:2], params['b_a'])]

            # compute the additive inputs
            u = [np.zeros(x[0].shape), *tree_map(compute_inputs, params['W_u'], x[:2])]

            # update the states
            x_new = tree_map(update_state, A, gamma, x, alphas, u)

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
            outputs = alphas, u, x_new, pen_xy_new, p_xy, pen_down_log_p

            return carry, outputs

        # initial LDS states
        # state of top layer is inferred by the encoder, states of lower layers are set to 0
        n_layers = len(self.x_dim)
        x0 = [z2[:], *[np.zeros(self.x_dim[layer]) for layer in range(1, n_layers)]]

        # initialise pen position at the one True pixel in the second channel of the image
        # pen position is in image coordinates (distance from top of image, distance from left of image)
        # pen_xy0 = self.image_dimensions / 2
        pen_xy0 = np.squeeze(np.array(np.where(data, size = 1))).astype(float)

        carry = x0, pen_xy0
        inputs = np.repeat(z1[None,:], self.T, axis = 0)

        _, (alphas, u, x, pen_xy, p_xy_t, pen_down_log_p) = lax.scan(decode_one_step, carry, inputs)
    
        return {'alphas': alphas,
                'u': u,
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
    tau: float

    def setup(self):
        
        self.encoder = encoder(self.n_loops_top_layer, self.x_dim[0])
        self.sampler = sampler(self.n_loops_top_layer)
        self.decoder = decoder(self.x_dim, self.image_dim, self.T, self.dt, self.tau)

    def __call__(self, data, params, A, gamma, key):

        output_encoder = self.encoder(data[None,:,:,:])
        z1, z2 = self.sampler(output_encoder, params, key)
        output_decoder = self.decoder(data[:,:,1], params, A, gamma, z1, z2)
        
        return output_encoder | output_decoder
