from train import create_train_state
from flax.training import checkpoints

def reload_state(model, init_params, args):

    state, _ = create_train_state(model, init_params, args)
    restored_state = checkpoints.restore_checkpoint(ckpt_dir = 'runs/' + args.folder_name, target = state)

    return restored_state