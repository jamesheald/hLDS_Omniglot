from jax import random, value_and_grad, vmap, jit
import jax.numpy as np
import numpy as onp
from jax.tree_util import tree_map
import optax
from flax.training import train_state #, checkpoints
from flax.training.early_stopping import EarlyStopping
import tensorflow_datasets as tfds
from utils import keyGen, stabilise_variance, smooth_maximum, print_metrics, write_images_to_tensorboard, write_metrics_to_tensorboard, save_best_checkpoints
from initialise import construct_dynamics_matrix
from flax.metrics import tensorboard
import time
from copy import copy
import jax
from jax import lax

def kl_scheduler(args):
    
    kl_schedule = []
    for i_batch in range(int(args.n_batches_train * args.n_epochs)):
        
        warm_up_fraction = min(max((i_batch - args.kl_warmup_start) / (args.kl_warmup_end - args.kl_warmup_start), 0), 1)
        
        kl_schedule.append(args.kl_min + warm_up_fraction * (args.kl_max - args.kl_min))

    return iter(kl_schedule)

def get_train_state(model, params, args):
    
    lr_scheduler = optax.exponential_decay(args.step_size, args.decay_steps, args.decay_factor)

    optimiser = optax.chain(optax.adamw(learning_rate = lr_scheduler, b1 = args.adam_b1, b2 = args.adam_b2, eps = args.adam_eps, 
                            weight_decay = args.weight_decay), optax.clip_by_global_norm(args.max_grad_norm))
    
    state = train_state.TrainState.create(apply_fn = model.apply, params = params, tx = optimiser)

    if args.reload_state:

        state = checkpoints.restore_checkpoint(ckpt_dir = 'runs/' + args.reload_folder_name, target = state)

    return state, lr_scheduler

def create_tensorboard_writer(args):

    # create a tensorboard writer
    # to view tensorboard results, call 'tensorboard --logdir=.' in runs folder from terminal
    writer = tensorboard.SummaryWriter('runs/' + args.folder_name)

    return writer

def apply_model(apply_fn, params, A, gamma, data, key):

    return apply_fn({'params': {'encoder': params['encoder']}}, data, params['decoder'], A, gamma, key)
    # return apply_fn({'params': state.params}, data, state.params['decoder'], A, gamma, key)

batch_apply_model = vmap(apply_model, in_axes = (None, None, None, None, 0, 0))
    
def loss_fn(params, state, data, kl_weight, key):

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
    output = batch_apply_model(state.apply_fn, params, A, gamma, data, subkeys)

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

    def breakpoint_if_nan(all_losses, output):
        is_nan = np.isnan(all_losses['total']).any()
        def true_fn(operands):
            jax.debug.breakpoint()
        def false_fn(operands):
            all_losses, output = operands
            pass
        lax.cond(is_nan, true_fn, false_fn, (all_losses, output))

    breakpoint_if_nan(all_losses, output)

    return all_losses['total'], (all_losses, output)

loss_grad = value_and_grad(loss_fn, has_aux = True)
eval_step_jit = jit(loss_fn)

def train_step(state, training_data, kl_weight, key):
    
    (loss, (all_losses, _)), grads = loss_grad(state.params, state, training_data, kl_weight, key)

    state = state.apply_gradients(grads = grads)
    
    return state, all_losses

train_step_jit = jit(train_step)

def optimise_model(model, params, train_dataset, validate_dataset, args, key):

    # start optimisation timer
    optimisation_start_time = time.time()

    # create schedule for KL divergence weight
    kl_schedule = kl_scheduler(args)

    # create train state
    state, lr_scheduler = get_train_state(model, params, args)

    # set early stopping criteria
    early_stop = EarlyStopping(min_delta = args.min_delta, patience = args.patience)

    # create tensorboard writer
    writer = create_tensorboard_writer(args)

    # initialise the validate losses for the n best checkpoints (initialise to infinite values that will be easily bettered/overwritten)
    best_validate_losses = onp.ones((args.n_best_checkpoints)) * np.inf

    # loop over epochs
    for epoch in range(args.n_epochs):
        
        # start epoch timer
        epoch_start_time = time.time()
        
        # convert the tf.data.Dataset train_dataset into an iterable
        # this iterable is shuffled differently each epoch
        train_datagen = iter(tfds.as_numpy(train_dataset))

        # generate subkeys
        key, training_subkeys = keyGen(key, n_subkeys = args.n_batches_train)

        # initialise the training losses and the timer
        training_losses = {'total': 0, 'cross_entropy': 0, 'kl': 0, 'kl_prescale': 0}
        batch_start_time = time.time()

        # loop over batches
        for batch in range(1, args.n_batches_train + 1):

            kl_weight = np.array(next(kl_schedule))

            state, all_losses = train_step_jit(state, np.array(next(train_datagen)['image']), kl_weight, next(training_subkeys))

            # training losses (average of 'print_every' batches)
            training_losses = tree_map(lambda x, y: x + y / args.print_every, training_losses, all_losses)

            if batch % args.print_every == 0:

                # end batches timer
                batches_duration = time.time() - batch_start_time

                # print metrics
                print_metrics("batch", batches_duration, training_losses, batch_range = [batch - args.print_every + 1, batch], 
                              lr = lr_scheduler(state.step - 1))

                # store losses
                if batch == args.print_every:
                    
                    t_losses_thru_training = copy(training_losses)
                    
                else:
                    
                    t_losses_thru_training = tree_map(lambda x, y: np.append(x, y), t_losses_thru_training, training_losses)

                # re-initialise the losses and timer
                training_losses = {'total': 0, 'cross_entropy': 0, 'kl': 0, 'kl_prescale': 0}
                batch_start_time = time.time()

        # calculate loss on validation data
        validate_datagen = iter(tfds.as_numpy(validate_dataset))

        # generate subkeys
        key, validate_subkeys = keyGen(key, n_subkeys = args.n_batches_validate)

        # initialise the losses
        validate_losses = {'total': 0, 'cross_entropy': 0, 'kl': 0, 'kl_prescale': 0}

        # loop over batches
        for batch in range(1, args.n_batches_validate + 1):

            v_data = np.array(next(validate_datagen)['image'])

            _, (all_losses, output) = eval_step_jit(state.params, state, v_data, kl_weight, next(validate_subkeys))

            # store validate losses
            validate_losses = tree_map(lambda x, y: x + y / args.n_batches_validate, validate_losses, all_losses)

        # calculate loss on validation data
        # key, validation_subkeys = keyGen(key, n_subkeys = 1)
        # _, (validation_losses, output) = eval_step_jit(state.params, state, validate_dataset, kl_weight, next(validation_subkeys))
            
        # end epoch timer
        epoch_duration = time.time() - epoch_start_time

        # print losses (mean over all batches in epoch)
        # print_metrics("epoch", epoch_duration, t_losses_thru_training, validation_losses, epoch = epoch + 1)
        print_metrics("epoch", epoch_duration, t_losses_thru_training, validate_losses, epoch = epoch + 1)

        if epoch % args.checkpoint_every == 0:

            # write images to tensorboard
            # write_images_to_tensorboard(writer, output, args, validate_dataset, epoch)
            write_images_to_tensorboard(writer, output, args, v_data, epoch)

            # write metrics to tensorboard
            # write_metrics_to_tensorboard(writer, t_losses_thru_training, validation_losses, epoch)
            write_metrics_to_tensorboard(writer, t_losses_thru_training, validate_losses, epoch)
                
            # save checkpoint
            # checkpoints.save_checkpoint(ckpt_dir = 'runs/' + args.folder_name, target = state, step = epoch)
        
        best_validate_losses = save_best_checkpoints(state, args, epoch, validate_losses['total'], best_validate_losses)

        # if early stopping criteria met, break
        # _, early_stop = early_stop.update(validation_losses['total'].mean())
        #_, early_stop = early_stop.update(validate_losses['total'].mean())
        #if early_stop.should_stop:
        #        
        #    print('Early stopping criteria met, breaking...')
        #        
        #    break

    optimisation_duration = time.time() - optimisation_start_time

    print('Optimisation finished in {:.2f} hours.'.format(optimisation_duration / 60**2))
            
    return state
