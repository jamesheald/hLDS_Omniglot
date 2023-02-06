from jax import random, value_and_grad, vmap, jit
import jax.numpy as np
import numpy as onp
from jax.tree_util import tree_map
import optax
from flax.training import checkpoints, train_state
from flax.training.early_stopping import EarlyStopping
import tensorflow_datasets as tfds
from utils import keyGen, stabilise_variance, smooth_maximum, print_metrics, write_images_to_tensorboard, write_metrics_to_tensorboard
from initialise import construct_dynamics_matrix
from flax.metrics import tensorboard
import time
from copy import copy
import jax
from jax import lax

def kl_scheduler(args):
    
    kl_schedule = []
    for i_batch in range(int(args.n_batches * args.n_epochs)):
        
        warm_up_fraction = min(max((i_batch - args.kl_warmup_start) / (args.kl_warmup_end - args.kl_warmup_start), 0), 1)
        
        kl_schedule.append(args.kl_min + warm_up_fraction * (args.kl_max - args.kl_min))

    return iter(kl_schedule)

def get_train_state(models, params, args, model_label):

    if model_label == 'vae':

        model = models[0]
        param = params[0]

    elif model_label == 'myosuite':

        model = models[1]
        param = params[1]
    
    lr_scheduler = optax.exponential_decay(args.step_size, args.decay_steps, args.decay_factor)

    optimiser = optax.chain(optax.adamw(learning_rate = lr_scheduler, b1 = args.adam_b1, b2 = args.adam_b2, eps = args.adam_eps, 
                            weight_decay = args.weight_decay), optax.clip_by_global_norm(args.max_grad_norm))
    
    state = train_state.TrainState.create(apply_fn = model.apply, params = param, tx = optimiser)

    if args.reload_state:

        state = checkpoints.restore_checkpoint(ckpt_dir = 'runs/' + args.reload_folder_name + '_' + model_label, target = state)

    return state, lr_scheduler

def create_tensorboard_writer(args):

    # create a tensorboard writer
    # to view tensorboard results, call 'tensorboard --logdir=.' in runs folder from terminal
    writer = tensorboard.SummaryWriter('runs/' + args.folder_name)

    return writer

def apply_model(apply_fn, params, data, A, gamma, state_myo, key):

    return apply_fn({'params': {'encoder': params['encoder']}}, data, params['decoder'], A, gamma, state_myo, key)

batch_apply_model = vmap(apply_model, in_axes = (None, None, None, None, 0, 0))
    
def loss_fn(params, state, state_myo, data, kl_weight, key):

    def cross_entropy_loss(p_xy_t, data):
    
        # compute the smooth maximum of the per-pixel bernoulli parameter across time steps
        p_xy = smooth_maximum(p_xy_t)

        # compute the logit for each pixel
        logits = np.log(p_xy / (1 - p_xy))

        # compute the total cross entropy across pixels
        cross_entropy = np.sum(optax.sigmoid_binary_cross_entropy(logits, data))

        return cross_entropy

    def KL_diagonal_Gaussians(mu_0, log_var_0, mu_1, log_var_1):
        """
        KL(q||p), where q is posterior and p is prior
        mu_0, log_var_0 is the mean and log variances of the prior
        mu_1, log_var_1 is the mean and log variances of the posterior
        var_min is added to the variances for numerical stability
        """
        log_var_0 = stabilise_variance(log_var_0)
        log_var_1 = stabilise_variance(log_var_1)

        return np.sum(0.5 * (log_var_0 - log_var_1 + np.exp(log_var_1 - log_var_0) 
                             - 1.0 + (mu_1 - mu_0)**2 / np.exp(log_var_0)))

    batch_cross_entropy_loss = vmap(cross_entropy_loss, in_axes = (0, 0))
    batch_KL_diagonal_Gaussians = vmap(KL_diagonal_Gaussians, in_axes = (None, None, 0, 0))

    # construct the dynamics of each loop from the parameters
    A, gamma = construct_dynamics_matrix(params['decoder'])
    
    # create a subkey for each example in the batch
    batch_size = data.shape[0]
    subkeys = random.split(key, batch_size)

    # apply the model
    output = batch_apply_model(state.apply_fn, params, data, A, gamma, state_myo, subkeys)

    # calculate the cross entropy
    cross_entropy = batch_cross_entropy_loss(output['p_xy_t'], data).mean()

    # calculate the KL divergence between the approximate posterior and prior over the latents
    mu_0 = 0
    log_var_0 = params['prior_z_log_var']
    mu_1 = output['z_mean']
    log_var_1 = output['z_log_var']
    kl_loss_prescale = batch_KL_diagonal_Gaussians(mu_0, log_var_0, mu_1, log_var_1).mean()
    kl_loss = kl_weight * kl_loss_prescale

    all_losses = {'cross_entropy': cross_entropy, 'kl': kl_loss, 'kl_prescale': kl_loss_prescale, 'total': cross_entropy + kl_loss}

    # def breakpoint_if_nan(all_losses, output):
    #     is_nan = np.isnan(all_losses['total']).any()
    #     def true_fn(operands):
    #         jax.debug.breakpoint()
    #     def false_fn(operands):
    #         all_losses, output = operands
    #         pass
    #     lax.cond(is_nan, true_fn, false_fn, (all_losses, output))

    # breakpoint_if_nan(all_losses, output)

    return all_losses['total'], (all_losses, output)

loss_grad_vae = value_and_grad(loss_fn, has_aux = True)
eval_step_jit = jit(loss_fn)

def train_step_vae(state, state_myo, training_data, kl_weight, key):
    
    (loss, (all_losses, output)), grads = loss_grad_vae(state.params, state, state_myo, training_data, kl_weight, key)

    state = state.apply_gradients(grads = grads)
    
    return state, all_losses, output['muscle_inputs']

train_step_vae_jit = jit(train_step_vae)

def perform_myosuite_simulations(env, muscle_inputs)

    # number of characters
    n_characters = muscle_inputs.shape[0]
    
    # number of time steps
    n_timesteps = muscle_inputs.shape[1]

    # preallocate arrays for storing the results of the myosuite simulations (fingertip position)
    fingertip_xyz = onp.empty((n_examples, n_timesteps, 3))

    for character in range(n_characters):

        # reset the environment
        env.reset()

        for timestep in range(n_timesteps):

            # apply the muscle inputs and step forward in time
            # (pass a copy of the muscle inputs to env.step as it performs a sigmoid transformation)
            obs, reward, done, info = env.step(onp.copy(muscle_inputs[s,t,:]))
        
            # store the fingertip position
            fingertip_xyz[character,timestep,:] = onp.copy(env.get_obs_dict(env.sim)['tip_pos'])

    return env, np.array(fingertip_position)

def loss_fn_myo(params, state, muscle_inputs, fingertip_position):

    carry = x0, state.params['carry_init']
    inputs = np.repeat(z1[None,:], self.T, axis = 0)

    _, (alphas, u, x, muscle_inputs, fingertip_xyz, pen_xy, pen_down_log_p, p_xy_t) = lax.scan(decode_one_step, carry, inputs)

    loss = 

    return loss

loss_grad_myo = value_and_grad(loss_fn_myo, has_aux = True)

def train_step_myo(state_myo, muscle_inputs, fingertip_position):
    
    (loss, (all_losses, output)), grads = loss_grad_myo(state_myo.params, state_myo, muscle_inputs, fingertip_position)

    state = state.apply_gradients(grads = grads)
    
    return state, all_losses, output['muscle_inputs']

train_step_myo_jit = jit(train_step_myo)

def optimise_model(models, params, train_dataset, validate_dataset, args, key):

    # start optimisation timer
    optimisation_start_time = time.time()

    # create schedule for KL divergence weight
    kl_schedule = kl_scheduler(args)
    
    # create train states
    state_vae, lr_scheduler = get_train_state(models, params, args, 'vae')
    state_myo, lr_scheduler = get_train_state(models, params, args, 'myosuite')
    
    # set early stopping criteria
    early_stop = EarlyStopping(min_delta = args.min_delta, patience = args.patience)

    # create tensorboard writer
    writer = create_tensorboard_writer(args)
    
    # loop over epochs
    for epoch in range(args.n_epochs):
        
        # start epoch timer
        epoch_start_time = time.time()
        
        # convert the tf.data.Dataset train_dataset into an iterable
        # this iterable is shuffled differently each epoch
        train_datagen = iter(tfds.as_numpy(train_dataset))

        # generate subkeys
        key, training_subkeys = keyGen(key, n_subkeys = args.n_batches)

        # initialise the losses and the timer
        training_losses = {'total': 0, 'cross_entropy': 0, 'kl': 0, 'kl_prescale': 0}
        batch_start_time = time.time()

        # loop over batches
        for batch in range(1, args.n_batches + 1):

            kl_weight = np.array(next(kl_schedule))

            # train the vae while fixing the parameters of the myosuite dynamics model
            state_vae, all_losses, muscle_inputs = train_step_vae_jit(state_vae, state_myo, np.array(next(train_datagen)['image']), kl_weight, next(training_subkeys))

            # pass the sequence of muscle inputs produced by the decoder through myosuite to obtain a sequence of fingertip positions
            env, fingertip_position = perform_myosuite_simulations(env, muscle_inputs)

            # train the myosuite dynamics model
            state_vae, all_losses, muscle_inputs = train_step_myo_jit(state_vae, state_myo, np.array(next(train_datagen)['image']), kl_weight, next(training_subkeys))

            # training losses (average of 'print_every' batches)
            training_losses = tree_map(lambda x, y: x + y / args.print_every, training_losses, all_losses)

            #if batch % args.print_every == 0:

        if epoch % 500 == 0:

                # end batches timer
                batches_duration = time.time() - batch_start_time

                # print metrics
                print_metrics("batch", batches_duration, training_losses, batch_range = [batch - args.print_every + 1, batch], 
                              lr = lr_scheduler(state_vae.step - 1))

                # store losses
                if batch == args.print_every:
                    
                    t_losses_thru_training = copy(training_losses)
                    
                else:
                    
                    t_losses_thru_training = tree_map(lambda x, y: np.append(x, y), t_losses_thru_training, training_losses)

                # re-initialise the losses and timer
                training_losses = {'total': 0, 'cross_entropy': 0, 'kl': 0, 'kl_prescale': 0}
                batch_start_time = time.time()

        if epoch % 500 == 0:

            # calculate loss on validation data
            key, validation_subkeys = keyGen(key, n_subkeys = 1)
            _, (validation_losses, output) = eval_step_jit(state.params, state, validate_dataset, kl_weight, next(validation_subkeys))
            
            # end epoch timer
            epoch_duration = time.time() - epoch_start_time
            
            # print losses (mean over all batches in epoch)
            print_metrics("epoch", epoch_duration, t_losses_thru_training, validation_losses, epoch = epoch + 1)

            # write images to tensorboard
            write_images_to_tensorboard(writer, output, args, validate_dataset, epoch)

            # write metrics to tensorboard
            write_metrics_to_tensorboard(writer, t_losses_thru_training, validation_losses, epoch)
            
            # save checkpoint
            checkpoints.save_checkpoint(ckpt_dir = 'runs/' + args.folder_name + '_' + 'vae', target = state_vae, step = epoch)
            checkpoints.save_checkpoint(ckpt_dir = 'runs/' + args.folder_name + '_' + 'myosuite', target = state_myosuite, step = epoch)
        
            # # if early stopping criteria met, break
            # _, early_stop = early_stop.update(validation_losses['total'].mean())
            # if early_stop.should_stop:
                
            #     print('Early stopping criteria met, breaking...')
                
            #     break

    optimisation_duration = time.time() - optimisation_start_time

    print('Optimisation finished in {:.2f} hours.'.format(optimisation_duration / 60**2))
            
    return
