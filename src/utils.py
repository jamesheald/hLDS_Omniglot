from jax import random, vmap
import jax.numpy as np
from flax import linen as nn
from jax.tree_util import tree_map
from moviepy.video.io.bindings import mplfig_to_npimage
from skimage import color
import matplotlib.pyplot as plt
import gym
import myosuite

def construct_dynamics_matrix(params):

	def construct_P(P):
		return P @ P.T

	def construct_S(U, V):
		return U @ np.transpose(V, (0, 2, 1)) - V @ np.transpose(U, (0, 2, 1))

	def construct_A(L, P, S):
		return (-L @ np.transpose(L, (0, 2, 1)) + S) @ P

	def squash_gammas(gamma):
		return nn.sigmoid(gamma)

	# positive semi-definite matrix P
	P = tree_map(construct_P, params['P'])

	# skew symmetric matrix S
	S = tree_map(construct_S, params['S_U'], params['S_V'])

	# dynamics matrix A (loops organised along axis 0)
	A = tree_map(construct_A, params['L'], P, S)

	# mixing coefficient gamma (0,1)
	gamma = tree_map(squash_gammas, params['gamma'])

	return A, gamma

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

def bound_variable(x, x_centre, x_range):
    
    # subtract x_centre from x
    x = x - x_centre

    # transform x such that the bounds of x become -/+ f, where f determines the degree of squashing
    f = 2
    x = x / (x_range / 2) * f

    # squash x between x_centre - x_range / 2 and x_centre + x_range / 2
    x = x_centre + x_range * (nn.sigmoid(x) - 1 / 2)

    return x

def instantiate_a_myosuite_environment(args, key):

    # create an instance of the myosuite environment
    env = gym.make(args.myosuite_environment)

    # generate a random seed for the myosuite environment
    env_seed = get_environment_seed(key)
    env.seed(env_seed)

    # set the duration of each timestep of the myosuite simulation in seconds
    # this should be done before calling env.reset to ensure that env.sim.data.qvel is in the correct units
    env = set_the_timestep_duration(env, args.myosuite_dt)
    
    return env

def get_environment_seed(key):
    
    seed = int(random.randint(key, (), 0, np.iinfo(np.int32).max))
    
    return seed

def set_the_timestep_duration(env, dt):
    
    # mujoco calculates the timestep duration based on the frame_skip variable, so set frame_skip
    env.env.frame_skip = dt / env.sim.model.opt.timestep
        
    return env

def print_metrics(phase, duration, t_losses, v_losses = [], batch_range = [], lr = [], epoch = []):
	
	if phase == "batch":
		
		s1 = '\033[1m' + "Batches {}-{} in {:.2f} seconds, learning rate: {:.5f}" + '\033[0m'
		print(s1.format(batch_range[0], batch_range[1], duration, lr))
		
	elif phase == "epoch":
		
		s1 = '\033[1m' + "Epoch {} in {:.1f} minutes" + '\033[0m'
		print(s1.format(epoch, duration / 60))
		
	s2 = """  Training loss VAE {:.4f} = cross entropy {:.4f} + KL {:.4f} ({:.4f})"""
	print(s2.format(t_losses['total'].mean(), t_losses['cross_entropy'].mean(),
					t_losses['kl'].mean(), t_losses['kl_prescale'].mean()))

	s3 = """  Training loss myosuite {:.6f}"""
	print(s3.format(t_losses['mse'].mean()))

	if phase == "epoch":

		s4 = """  Validation loss VAE {:.4f} = cross entropy {:.4f} + KL {:.4f} ({:.4f})\n"""
		print(s4.format(v_losses['total'].mean(), v_losses['cross_entropy'].mean(),
						v_losses['kl'].mean(), v_losses['kl_prescale'].mean()))

def create_figure(args):

	fig, axs = plt.subplots(args.square_image_grid_size, args.square_image_grid_size)
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

def original_images(validate_dataset, args):

	# create figure of original images
	fig, axs = create_figure(args)
	for i, ax in enumerate(axs.ravel()):

		ax.imshow(validate_dataset[i,:,:], cmap = 'gray')
		ax.set_xticks([])
		ax.set_yticks([])

	# convert figure to 2D numpy array
	image = convert_figure_to_image(fig)

	return image

def reconstructed_images(pen_xy, pen_down_log_p, args):

	# create figure of reconstructed images
	fig, axs = create_figure(args)
	T = len(pen_xy[0][:,0])
	for i, ax in enumerate(axs.ravel()):

		for t in range(T - 1):

			ax.plot(pen_xy[i][t:t + 2,1], args.image_dim[0] - pen_xy[i][t:t + 2,0], alpha =  float(np.exp(pen_down_log_p[i,t])), color = 'k', linewidth = 5)

		ax.set_ylim([0, args.image_dim[0]])
		ax.set_xlim([0, args.image_dim[1]])
		ax.set_yticks([])
		ax.set_xticks([])

	# convert figure to 2D numpy array
	image = convert_figure_to_image(fig)

	return image

def write_images_to_tensorboard(writer, output, args, validate_dataset, epoch):

	if epoch == 0:

		writer.image("original_images", original_images(validate_dataset, args), epoch)

	writer.image("reconstructed_images", reconstructed_images(output['pen_xy'], output['pen_down_log_p'], args), epoch)

def write_metrics_to_tensorboard(writer, t_losses, v_losses, epoch):

	writer.scalar('VAE loss (train)', t_losses['total'].mean(), epoch)
	writer.scalar('cross entropy (train)', t_losses['cross_entropy'].mean(), epoch)
	writer.scalar('mse (train)', t_losses['mse'].mean(), epoch)
	writer.scalar('KL (train)', t_losses['kl'].mean(), epoch)
	writer.scalar('KL prescale (train)', t_losses['kl_prescale'].mean(), epoch)
	writer.scalar('VAE loss (validation)', v_losses['total'].mean(), epoch)
	writer.scalar('cross entropy (validation)', v_losses['cross_entropy'].mean(), epoch)
	writer.scalar('KL (validation)', v_losses['kl'].mean(), epoch)
	writer.scalar('KL prescale (validation)', v_losses['kl_prescale'].mean(), epoch)
	writer.flush()

def forward_pass_model(model_vae, params_vae, data, state_myo, args, key):

	def apply_model(model_vae, params_vae, data, A, gamma, state_myo, key):

		return model_vae.apply({'params': {'encoder': params_vae['params']['encoder']}}, data, params_vae['decoder'], A, gamma, state_myo, key)

	batch_apply_model = vmap(apply_model, in_axes = (None, None, 0, None, None, None, 0))

	# construct the dynamics of each loop from the parameters
	A, gamma = construct_dynamics_matrix(params_vae['decoder'])

	# create a subkey for each example in the batch
	batch_size = data.shape[0]
	subkeys = random.split(key, batch_size)

	# apply the model
	output = batch_apply_model(model_vae, params_vae, data, A, gamma, state_myo, subkeys)

	# store the original and reconstructed images in the model output
	output['input_images'] = original_images(data, args)
	output['output_images'] = reconstructed_images(output['pen_xy'], output['pen_down_log_p'], args)

	return output