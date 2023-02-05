import myosuite
import gym

def instantiate_a_myosuite_environment(hyperparams, environment_name, key):

    # create an instance of the myosuite environment
    env = gym.make(environment_name)

    # generate a random seed for the myosuite environment
    env_seed = get_environment_seed(key)
    env.seed(env_seed)

    # set the duration of each timestep of the myosuite simulation in seconds
    # this should be done before calling env.reset to ensure that env.sim.data.qvel is in the correct units
    env = set_the_timestep_duration(env, hyperparams['dt'])
    
    return env

def get_environment_seed(key):
    
    seed = int(random.randint(key, (), 0, np.iinfo(np.int32).max))
    
    return seed

def set_the_timestep_duration(env, dt):
    
    # mujoco calculates the timestep duration based on the frame_skip variable, so set frame_skip
    env.env.frame_skip = dt / env.sim.model.opt.timestep
        
    return env

def perform_myosuite_simulations(env, results, scaler):
    
    # preallocate arrays for storing the results of the myosuite simulations
    results['qpos_myo'] = onp.empty(results['qpos_mean'].shape)
    results['qvel_myo'] = onp.empty(results['qvel'].shape)
    results['act_myo'] = onp.empty(results['act'].shape)
    
    # number of sequences
    n_sequences = results['muscle_inputs'].shape[0]
    
    # number of time steps
    n_timesteps = results['muscle_inputs'].shape[1]
    
    # unstandardise the initial joint angles and joint angular velocities to pass to myosuite
    results['qpos0'] = transform_data(scaler, results['qpos0'], 'inverse_transform')
    results['qvel0'] = transform_data(scaler, results['qvel0'], 'inverse_transform')
    
    for s in range(n_sequences):

        # reset myosuite
        env.reset()

        # initialise the state of myosuite based on encoder inferences
        # n.b. env.sim.data.qvel is in radians/second whereas results['qvel0'] is in radians/dt
        env.sim.data.qpos[:] = onp.copy(results['qpos0'][s,:])
        env.sim.data.qvel[:] = onp.copy(results['qvel0'][s,:]/env.dt)
        env.sim.data.act[:] = onp.copy(results['act0'][s,:])

        for t in range(n_timesteps):

            # apply the muscle inputs and step forward in time
            # (pass a copy of the muscle inputs to env.step as it performs a sigmoid transformation)            
            obs, reward, done, info = env.step(onp.copy(results['muscle_inputs'][s,t,:]))
        
            # assign the myosuite
            results['qpos_myo'][s,t,:] = onp.copy(env.get_obs_dict(env.sim)['qpos'])
            results['qvel_myo'][s,t,:] = onp.copy(env.get_obs_dict(env.sim)['qvel'])
            results['act_myo'][s,t,:] = onp.copy(env.get_obs_dict(env.sim)['act'])
    
    # standardise the myosuite joint angles and joint angular velocities based on the behavioural training data
    results['qpos_myo'] = transform_data(scaler, results['qpos_myo'], 'transform')
    results['qvel_myo'] = transform_data(scaler, results['qvel_myo'], 'transform')
    
    return env, results
