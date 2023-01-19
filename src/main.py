import argparse
import pickle
import os

from load_data import create_data_split
from initialise import initialise_model
from train import optimise_model

def main():

    parser = argparse.ArgumentParser(description = 'hyperparameters')

    # model
    parser.add_argument('--folder_name',             default = 'hLDS')

    # model
    parser.add_argument('--x_dim',                   default = [20, 50, 200])
    parser.add_argument('--alpha_fraction',          type = float, default = 0.1)
    parser.add_argument('--dt',                      type = float, default = 0.01)
    parser.add_argument('--tau',                     type = float, default = 0.2)
    parser.add_argument('--time_steps',              type = float, default = 100)
    parser.add_argument('--init_pen_log_var',        type = float, default = 5.0) # 5.7 correponds to 3 SD = 105/2 (canvas half width)
    parser.add_argument('--image_dim',               default = [105, 105])
    parser.add_argument('--prior_z_log_var',         type = float, default = -2.0)
    parser.add_argument('--jax_seed',                type = int, default = 0)

    # data
    parser.add_argument('--percent_data_to_use',     type = int, default = 100)
    parser.add_argument('--fraction_for_validation', type = int, default = 0.2)
    parser.add_argument('--batch_size',              type = int, default = 32)
    parser.add_argument('--data_seed',               type = int, default = 0)

    # images to write to tensorboard
    parser.add_argument('--square_image_grid_size',  type = int, default = 2)

    # optimisation
    parser.add_argument('--kl_warmup_start',         type = int, default = 500)
    parser.add_argument('--kl_warmup_end',           type = int, default = 1000)
    parser.add_argument('--kl_min',                  type = float, default = 0.1) # 0.01
    parser.add_argument('--kl_max',                  type = float, default = 1.0)
    parser.add_argument('--adam_b1',                 type = float, default = 0.9)
    parser.add_argument('--adam_b2',                 type = float, default = 0.999)
    parser.add_argument('--adam_eps',                type = float, default = 1e-8)
    parser.add_argument('--weight_decay',            type = float, default = 0.0001)
    parser.add_argument('--max_grad_norm',           type = float, default = 10.0)
    parser.add_argument('--step_size',               type = float, default = 0.001)
    parser.add_argument('--decay_steps',             type = int, default = 1)
    parser.add_argument('--decay_factor',            type = float, default = 0.9999)
    parser.add_argument('--print_every',             type = int, default = 1)
    parser.add_argument('--n_epochs',                type = int, default = 10000)
    parser.add_argument('--min_delta',               type = float, default = 1e-3)
    parser.add_argument('--patience',                type = int, default = 2)

    args = parser.parse_args()

    # to change an argument via the command line: python main.py --folder_name 'run_1'

    # save the hyperparameters
    path = 'runs/' + args.folder_name + '/hyperparameters'
    os.makedirs(os.path.dirname(path))
    file = open(path, 'wb') # change 'wb' to 'rb' to load
    pickle.dump(args, file) # change to args = pickle.load(file) to load
    file.close()

    print('eventually remove if epoch % 50 == 0 code')

    train_dataset, validate_dataset = create_data_split(args)

    # from jax.config import config
    # config.update("jax_debug_nans", False)
    # config.update("jax_disable_jit", True)
    # type help at a breakpoint() to see available commands
    # use xeus-python kernel -- Python 3.9 (XPython) -- for debugging

    model, init_params, args, key = initialise_model(args, train_dataset)

    # optionally reload previously saved train state
    if False:

        from reload_checkpoint import reload_state
        args.folder_name = 'kl01_dx2_nobreak'
        state = reload_state(model, init_params, args)
        params = state.params

    # import jax
    # jax.profiler.start_trace('runs/' + folder_name)
    state = optimise_model(model, init_params, train_dataset, validate_dataset, args, key)
    # jax.profiler.stop_trace()

    # # train_dataset = np.array(list(tfds.as_numpy(train_dataset))[0]['image']).reshape(args.batch_size,105,105,1)
    # from utils import forward_pass_model
    # params = init_params
    # output = forward_pass_model(model, params, train_dataset, args, key)

if __name__ == '__main__':

    main()