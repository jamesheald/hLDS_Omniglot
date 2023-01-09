from jax import random, vmap
import jax.numpy as np
from flax import linen as nn
from jax.tree_util import tree_map
from moviepy.video.io.bindings import mplfig_to_npimage
from skimage import color
import matplotlib.pyplot as plt

def construct_dynamics_matrix(params):

    def construct_P(P):
        return P @ P.T

    def construct_S(U, V):
        return U @ np.transpose(V, (0, 2, 1)) - V @ np.transpose(U, (0, 2, 1))

    def construct_A(L, P, S):
        return (-L @ np.transpose(L, (0, 2, 1)) + S) @ P

    # positive semi-definite matrix P
    P = tree_map(construct_P, params['P'])

    # skew symmetric matrix S
    S = tree_map(construct_S, params['S_U'], params['S_V'])

    # dynamics matrix A (loops organised along axis 0)
    A = tree_map(construct_A, params['L'], S, P)

    return A

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

def print_metrics(phase, duration, t_losses, v_losses = [], batch_range = [], lr = [], epoch = []):
    
    if phase == "batch":
        
        s1 = '\033[1m' + "Batches {}-{} in {:.2f} seconds, learning rate: {:.5f}" + '\033[0m'
        print(s1.format(batch_range[0], batch_range[1], duration, lr))
        
    elif phase == "epoch":
        
        s1 = '\033[1m' + "Epoch {} in {:.1f} minutes" + '\033[0m'
        print(s1.format(epoch, duration / 60))
        
    s2 = """  Training loss {:.4f} = cross entropy {:.4f} + KL {:.4f} ({:.4f})"""
    print(s2.format(t_losses['total'].mean(), t_losses['cross_entropy'].mean(),
                    t_losses['kl'].mean(), t_losses['kl_prescale'].mean()))

    if phase == "epoch":

        s3 = """  Validation loss {:.4f} = cross entropy {:.4f} + KL {:.4f} ({:.4f})\n"""
        print(s3.format(v_losses['total'].mean(), v_losses['cross_entropy'].mean(),
                        v_losses['kl'].mean(), v_losses['kl_prescale'].mean()))

def create_figure(cfg):

    fig, axs = plt.subplots(cfg.square_image_grid_size, cfg.square_image_grid_size)
    fig.figsize = (10, 10)
    fig.set_dpi(50)

    return fig, axs

def convert_figure_to_image(fig):

    # convert figure to 2D numpy array
    numpy_fig = mplfig_to_npimage(fig)
    gray_image = color.rgb2gray(numpy_fig)

    # close plot
    plt.close()

    return gray_image

def original_images(validate_dataset, cfg):

    # create figure of original images
    fig, axs = create_figure(cfg)
    for i, ax in enumerate(axs.ravel()):

        ax.imshow(validate_dataset[i,:,:], cmap = 'gray')
        ax.set_xticks([])
        ax.set_yticks([])

    # convert figure to 2D numpy array
    image = convert_figure_to_image(fig)

    return image

def reconstructed_images(pen_xy, pen_down_log_p, cfg):

    # create figure of reconstructed images
    fig, axs = create_figure(cfg)
    T = len(pen_xy[0][:,0])
    for i, ax in enumerate(axs.ravel()):

        # this assumes that the pen position is initialised in the middle of the canvas
        ax.plot([cfg.image_dim[1]/2, pen_xy[i][0,0]], [cfg.image_dim[0]/2, pen_xy[i][0,1]], alpha =  float(np.exp(pen_down_log_p[i,0])), color = 'k', linewidth = 5)

        for t in range(T - 1):

            ax.plot(pen_xy[i][t:t + 2,0], pen_xy[i][t:t + 2,1], alpha =  float(np.exp(pen_down_log_p[i,t + 1])), color = 'k', linewidth = 5)

        ax.set_xlim([0, cfg.image_dim[1]])
        ax.set_ylim([0, cfg.image_dim[0]])
        ax.set_xticks([])
        ax.set_yticks([])

    # convert figure to 2D numpy array
    image = convert_figure_to_image(fig)

    return image

def write_images_to_tensorboard(writer, pen_xy, pen_down_log_p, cfg, validate_dataset, epoch):

    if epoch == 0:

        writer.image("original_images", original_images(validate_dataset, cfg), epoch)

    writer.image("reconstructed_images", reconstructed_images(pen_xy, pen_down_log_p, cfg), epoch)

def write_metrics_to_tensorboard(writer, t_losses, v_losses, epoch):

    writer.scalar('loss (train)', t_losses['total'].mean(), epoch)
    writer.scalar('cross entropy (train)', t_losses['cross_entropy'].mean(), epoch)
    writer.scalar('KL (train)', t_losses['kl'].mean(), epoch)
    writer.scalar('KL prescale (train)', t_losses['kl_prescale'].mean(), epoch)
    writer.scalar('loss (validation)', v_losses['total'].mean(), epoch)
    writer.scalar('cross entropy (validation)', v_losses['cross_entropy'].mean(), epoch)
    writer.scalar('KL (validation)', v_losses['kl'].mean(), epoch)
    writer.scalar('KL prescale (validation)', v_losses['kl_prescale'].mean(), epoch)
    writer.flush()

def forward_pass_model(model, params, data, cfg, key):

    def apply_model(model, params, data, A, key):

        return model.apply({'params': {'encoder': params['encoder']}}, data, params['decoder'], A, key)

    batch_apply_model = vmap(apply_model, in_axes = (None, None, 0, None, 0))

    # construct the dynamics of each loop from the parameters
    A = construct_dynamics_matrix(params['decoder'])

    # create a subkey for each example in the batch
    batch_size = data.shape[0]
    subkeys = random.split(key, batch_size)

    # apply the model
    output = batch_apply_model(model, params, data, A, subkeys)

    output['input_images'] =  original_images(data, cfg)
    output['output_images'] = reconstructed_images(output['pen_xy'], output['pen_down_log_p'], cfg)

    return output