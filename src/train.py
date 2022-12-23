from jax import random, value_and_grad, vmap, jit
import jax.numpy as np
from jax.tree_util import tree_map
import optax
from flax.training import checkpoints, train_state
from flax.training.early_stopping import EarlyStopping
import tensorflow_datasets as tfds
from utils import keyGen, stabilise_variance, smooth_maximum
from initialise import construct_dynamics_matrix
import time
from copy import copy

def kl_scheduler(cfg):
    
    kl_schedule = []
    for i_batch in range(cfg.n_batches):
        
        warm_up_fraction = min(max((i_batch - cfg.kl_warmup_start) / (cfg.kl_warmup_end - cfg.kl_warmup_start), 0), 1)
        
        kl_schedule.append(cfg.kl_min + warm_up_fraction * (cfg.kl_max - cfg.kl_min))

    return iter(kl_schedule)

def create_train_state(model, init_params, cfg):
    
    lr_scheduler = optax.exponential_decay(cfg.step_size, cfg.decay_steps, cfg.decay_factor)

    optimiser = optax.chain(optax.adamw(learning_rate = lr_scheduler, b1 = cfg.adam_b1, b2 = cfg.adam_b2, eps = cfg.adam_eps, weight_decay = cfg.weight_decay), 
                            optax.clip_by_global_norm(cfg.max_grad_norm))
    
    state = train_state.TrainState.create(apply_fn = model.apply, params = init_params, tx = optimiser)

    return state, lr_scheduler

def apply_model(state, data, key, A, x0):

    return state.apply_fn({'params': {'encoder': state.params['encoder']}}, data, state.params['decoder'], key, A, x0)
    # return state.apply_fn({'params': state.params}, data, state.params['decoder'], key, A, x0)

batch_apply_model = vmap(apply_model, in_axes = (None, 0, 0, None, None))
    
def loss_fn(params, state, data, x0, kl_weight, key):

    def cross_entropy_loss(p_xy_t, data):
    
        # compute the smooth maximum of the per-pixel bernoulli parameter across time steps
        p_xy = smooth_maximum(p_xy_t)

        # compute the logit for each pixel
        logits = np.log(p_xy / (1 - p_xy))

        # compute the average cross entropy across pixels
        cross_entropy = np.mean(optax.sigmoid_binary_cross_entropy(logits, data))

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

    A = construct_dynamics_matrix(params['decoder'])
    
    # apply the model
    batch_size = data.shape[0]
    subkeys = random.split(key, batch_size)
    output = batch_apply_model(state, data, subkeys, A, x0)

    # calculate cross entropy
    cross_entropy = batch_cross_entropy_loss(output['p_xy_t'], data).mean()

    # calculate KL divergence between the approximate posterior and prior over the latents
    mu_0 = 0
    log_var_0 = params['prior_z_log_var']
    mu_1 = output['z_mean']
    log_var_1 = output['z_log_var']
    kl_loss_prescale = batch_KL_diagonal_Gaussians(mu_0, log_var_0, mu_1, log_var_1).mean()
    kl_loss = kl_weight * kl_loss_prescale

    loss = cross_entropy + kl_loss

    all_losses = {'total': loss, 'cross_entropy': cross_entropy, 'kl': kl_loss, 'kl_prescale': kl_loss_prescale}

    return loss, all_losses

loss_grad = value_and_grad(loss_fn, has_aux = True)
eval_step_jit = jit(loss_fn)

def train_step(state, training_data, x0, kl_weight, key):
    
    (loss, all_losses), grads = loss_grad(state.params, state, training_data, x0, kl_weight, key)

    state = state.apply_gradients(grads = grads)
    
    return state, loss, all_losses

train_step_jit = jit(train_step)

def print_metrics(phase, duration, t_losses, v_losses, batch_range = [], lr = [], epoch = []):
    
    if phase == "batch":
        
        s1 = '\033[1m' + "Batches {}-{} in {:.2f} seconds, learning rate: {:.5f}" + '\033[0m'
        print(s1.format(batch_range[0], batch_range[1], duration, lr))
        
    elif phase == "epoch":
        
        s1 = '\033[1m' + "Epoch {} in {:.1f} minutes" + '\033[0m'
        print(s1.format(epoch, duration / 60))
        
    s2 = """  Training losses {:.4f} = cross entropy {:.4f} + KL {:.4f} ({:.4f})"""
    s3 = """  Validation losses {:.4f} = cross entropy {:.4f} + KL {:.4f} ({:.4f})"""
    s3 = """  Validation losses {:.4f} = cross entropy {:.4f} + KL {:.4f} ({:.4f})"""
    print(s2.format(t_losses['total'].mean(), t_losses['cross_entropy'].mean(),
                    t_losses['kl'].mean(), t_losses['kl_prescale'].mean()))
    print(s3.format(v_losses['total'].mean(), v_losses['cross_entropy'].mean(),
                    v_losses['kl'].mean(), v_losses['kl_prescale'].mean()))
    
    if phase == "epoch":
        print("""\n""")
        
def write_to_tensorboard(writer, t_losses, v_losses, epoch):

    writer.scalar('loss (train)', t_losses['total'].mean(), epoch)
    writer.scalar('cross entropy (train)', t_losses['cross_entropy'].mean(), epoch)
    writer.scalar('KL (train)', t_losses['kl'].mean(), epoch)
    writer.scalar('KL prescale (train)', t_losses['kl_prescale'].mean(), epoch)
    writer.scalar('loss (validation)', v_losses['total'].mean(), epoch)
    writer.scalar('cross entropy (validation)', v_losses['cross_entropy'].mean(), epoch)
    writer.scalar('KL (validation)', v_losses['kl'].mean(), epoch)
    writer.scalar('KL prescale (validation)', v_losses['kl_prescale'].mean(), epoch)
    writer.flush()

def optimise_model(init_params, x0, model, train_dataset, validate_dataset, cfg, key, ckpt_dir, writer):

    kl_schedule = kl_scheduler(cfg)
    
    state, lr_scheduler = create_train_state(model, init_params, cfg)
    
    # set early stopping criteria
    early_stop = EarlyStopping(min_delta = cfg.min_delta, patience = cfg.patience)
    
    # loop over epochs
    losses = {}
    for epoch in range(cfg.n_epochs):
        
        # start epoch timer
        epoch_start_time = time.time()
        
        # convert the tf.data.Dataset train_dataset into an iterable
        # this iterable is shuffled differently each epoch
        train_datagen = iter(tfds.as_numpy(train_dataset))

        # generate subkeys
        key, training_subkeys = keyGen(key, n_subkeys = cfg.n_batches)
        key, validation_subkeys = keyGen(key, n_subkeys = int(cfg.n_batches / cfg.print_every))

        # initialise the losses and the timer
        training_losses = {'total': 0, 'cross_entropy': 0, 'kl': 0, 'kl_prescale': 0}
        batch_start_time = time.time()

        # loop over batches
        for batch in range(1, cfg.n_batches + 1):

            kl_weight = np.array(next(kl_schedule))

            state, loss, all_losses = train_step_jit(state, np.array(next(train_datagen)['image']), x0, kl_weight, next(training_subkeys))

            # training losses (average of 'print_every' batches)
            training_losses = tree_map(lambda x, y: x + y / cfg.print_every, training_losses, all_losses)

            if batch % cfg.print_every == 0:

                # calculate loss on validation data
                # _, validation_losses = eval_step_jit(state.params, state, validate_dataset, x0, kl_weight, next(validation_subkeys))
                validation_losses = training_losses
                    
                # end batches timer
                batches_duration = time.time() - batch_start_time

                # print metrics
                print_metrics("batch", batches_duration, training_losses, training_losses, 
                              batch_range = [batch - cfg.print_every + 1, batch], lr = lr_scheduler(batch - 1 + epoch * cfg.n_batches))

                # store losses
                if batch == cfg.print_every:
                    
                    t_losses_thru_training = copy(training_losses)
                    v_losses_thru_training = copy(validation_losses)
                    
                else:
                    
                    t_losses_thru_training = tree_map(lambda x, y: np.append(x, y), t_losses_thru_training, training_losses)
                    v_losses_thru_training = tree_map(lambda x, y: np.append(x, y), v_losses_thru_training, validation_losses)

                # re-initialise the losses and timer
                training_losses = {'total': 0, 'cross_entropy': 0, 'kl': 0, 'kl_prescale': 0}
                batch_start_time = time.time()

        losses['epoch ' + str(epoch)] = {'t_losses' : t_losses_thru_training, 'v_losses' : v_losses_thru_training}
        
        # end epoch timer
        epoch_duration = time.time() - epoch_start_time
        
        # print losses (mean over all batches in epoch)
        print_metrics("epoch", epoch_duration, t_losses_thru_training, v_losses_thru_training, epoch = epoch + 1)

        # write metrics to tensorboard
        write_to_tensorboard(writer, t_losses_thru_training, v_losses_thru_training, epoch)
        
        # save checkpoint
        ckpt = {'train_state': state, 'losses': losses, 'cfg': cfg}
        checkpoints.save_checkpoint(ckpt_dir = ckpt_dir, target = ckpt, step = epoch)
        
        # if early stopping criteria met, break
        _, early_stop = early_stop.update(v_losses_thru_training['total'].mean())
        if early_stop.should_stop:
            
            print('Early stopping criteria met, breaking...')
            
            break
            
    return state, losses