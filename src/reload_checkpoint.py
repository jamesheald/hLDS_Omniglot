from train import create_train_state
from flax.training import checkpoints

def reload_state(model, init_params, args, ckpt_dir):

    state, _ = create_train_state(model, init_params, args)
    ckpt = {'train_state': state}
    restored_state = checkpoints.restore_checkpoint(ckpt_dir = 'runs/' + args.folder_name, target = ckpt)

    return restored_state