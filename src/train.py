from jax import random, value_and_grad, vmap, jit
import jax.numpy as np
from jax.tree_util import tree_map
import optax
from flax.training import checkpoints, train_state
from flax.training.early_stopping import EarlyStopping
import tensorflow_datasets as tfds
from utils import keyGen, stabilise_variance, smooth_maximum, print_metrics, write_images_to_tensorboard, write_metrics_to_tensorboard
from initialise import construct_dynamics_matrix
import time
from copy import copy

def kl_scheduler(cfg):
    
    kl_schedule = []
    for i_batch in range(int(cfg.n_batches * cfg.n_epochs)):
        
        warm_up_fraction = min(max((i_batch - cfg.kl_warmup_start) / (cfg.kl_warmup_end - cfg.kl_warmup_start), 0), 1)
        
        kl_schedule.append(cfg.kl_min + warm_up_fraction * (cfg.kl_max - cfg.kl_min))

    return iter(kl_schedule)

def create_train_state(model, init_params, cfg):
    
    lr_scheduler = optax.exponential_decay(cfg.step_size, cfg.decay_steps, cfg.decay_factor)

    optimiser = optax.chain(optax.adamw(learning_rate = lr_scheduler, b1 = cfg.adam_b1, b2 = cfg.adam_b2, eps = cfg.adam_eps, 
                            weight_decay = cfg.weight_decay), optax.clip_by_global_norm(cfg.max_grad_norm))
    
    state = train_state.TrainState.create(apply_fn = model.apply, params = init_params, tx = optimiser)

    return state, lr_scheduler

def apply_model(state, data, A, gamma, key):

    return state.apply_fn({'params': {'encoder': state.params['encoder']}}, data, state.params['decoder'], A, gamma, key)
    # return state.apply_fn({'params': state.params}, data, state.params['decoder'], A, key)

batch_apply_model = vmap(apply_model, in_axes = (None, 0, None, None, 0))
    
def loss_fn(params, state, data, kl_weight, key):

    def cross_entropy_loss(p_xy_t, data):
    
        # compute the smooth maximum of the per-pixel bernoulli parameter across time steps
        p_xy = smooth_maximum(p_xy_t)

        # compute the logit for each pixel
        logits = np.log(p_xy / (1 - p_xy))

        # compute the total cross entropy across pixels
        cross_entropy = np.sum(optax.sigmoid_binary_cross_entropy(logits, data[:,:,0]))

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
    output = batch_apply_model(state, data, A, gamma, subkeys)

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

    return all_losses['total'], (all_losses, output)

loss_grad = value_and_grad(loss_fn, has_aux = True)
eval_step_jit = jit(loss_fn)

def train_step(state, training_data, kl_weight, key):
    
    (loss, (all_losses, _)), grads = loss_grad(state.params, state, training_data, kl_weight, key)

    state = state.apply_gradients(grads = grads)
    
    return state, all_losses

train_step_jit = jit(train_step)

def optimise_model(model, init_params, train_dataset, validate_dataset, cfg, key, ckpt_dir, writer):

    # start optimisation timer
    optimisation_start_time = time.time()

    # create schedule for KL divergence weight
    kl_schedule = kl_scheduler(cfg)
    
    # create train state
    state, lr_scheduler = create_train_state(model, init_params, cfg)
    
    # set early stopping criteria
    early_stop = EarlyStopping(min_delta = cfg.min_delta, patience = cfg.patience)
    
    # loop over epochs
    for epoch in range(cfg.n_epochs):
        
        # start epoch timer
        epoch_start_time = time.time()
        
        # convert the tf.data.Dataset train_dataset into an iterable
        # this iterable is shuffled differently each epoch
        # train_datagen = iter(tfds.as_numpy(train_dataset)) TFDS change

        # generate subkeys
        key, training_subkeys = keyGen(key, n_subkeys = cfg.n_batches)

        # initialise the losses and the timer
        training_losses = {'total': 0, 'cross_entropy': 0, 'kl': 0, 'kl_prescale': 0}
        batch_start_time = time.time()

        # loop over batches
        for batch in range(1, cfg.n_batches + 1):

            kl_weight = np.array(next(kl_schedule))

            # state, all_losses = train_step_jit(state, np.array(next(train_datagen)['image']), kl_weight, next(training_subkeys)) TFDS change
            state, all_losses = train_step_jit(state, train_dataset, kl_weight, next(training_subkeys))

            # training losses (average of 'print_every' batches)
            training_losses = tree_map(lambda x, y: x + y / cfg.print_every, training_losses, all_losses)

            if batch % cfg.print_every == 0:

                # end batches timer
                batches_duration = time.time() - batch_start_time

                # print metrics
                print_metrics("batch", batches_duration, training_losses, batch_range = [batch - cfg.print_every + 1, batch], 
                              lr = lr_scheduler(batch - 1 + epoch * cfg.n_batches))

                # store losses
                if batch == cfg.print_every:
                    
                    t_losses_thru_training = copy(training_losses)
                    
                else:
                    
                    t_losses_thru_training = tree_map(lambda x, y: np.append(x, y), t_losses_thru_training, training_losses)

                # re-initialise the losses and timer
                training_losses = {'total': 0, 'cross_entropy': 0, 'kl': 0, 'kl_prescale': 0}
                batch_start_time = time.time()

        # calculate loss on validation data
        key, validation_subkeys = keyGen(key, n_subkeys = 1)
        _, (validation_losses, output) = eval_step_jit(state.params, state, validate_dataset, kl_weight, next(validation_subkeys))
        
        # end epoch timer
        epoch_duration = time.time() - epoch_start_time
        
        # print losses (mean over all batches in epoch)
        print_metrics("epoch", epoch_duration, t_losses_thru_training, validation_losses, epoch = epoch + 1)

        # write images to tensorboard
        write_images_to_tensorboard(writer, output, cfg, validate_dataset, epoch)

        # write metrics to tensorboard
        write_metrics_to_tensorboard(writer, t_losses_thru_training, validation_losses, epoch)
        
        # save checkpoint
        ckpt = {'train_state': state}
        checkpoints.save_checkpoint(ckpt_dir = ckpt_dir, target = ckpt, step = epoch)
        
        # # if early stopping criteria met, break
        # _, early_stop = early_stop.update(validation_losses['total'].mean())
        # if early_stop.should_stop:
            
        #     print('Early stopping criteria met, breaking...')
            
        #     break

    optimisation_duration = time.time() - optimisation_start_time

    print('Optimisation finished in {:.2f} hours.'.format(optimisation_duration / 60**2))
            
    return state
