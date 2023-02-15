import tensorflow as tf
import tensorflow_datasets as tfds
import jax.numpy as np
# hide any GPUs from TensorFlow, otherwise TF might reserve memory and make it unavailable to JAX
tf.config.set_visible_devices([], device_type = 'GPU') # tf.config.experimental.set_visible_devices([], "GPU")

# https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html
# https://www.tensorflow.org/datasets/catalog/omniglot
# https://www.tensorflow.org/datasets/api_docs/python/tfds/load

def prepare_image(dictionary):
    
    # remove redundant channels
    dictionary['image'] = dictionary['image'][:,:,0]
    
    # invert image so drawn pixels are 1
    dictionary['image'] = tf.cast(dictionary['image'] == 0, tf.float32)

    return dictionary

def transform_train_dataset(dataset, batch_size, tfds_seed):
    
    dataset = dataset.cache()
    dataset = dataset.shuffle(tf.data.experimental.cardinality(dataset).numpy(), seed = tfds_seed, reshuffle_each_iteration = True)
    dataset = dataset.batch(batch_size, drop_remainder = True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

def transform_validate_dataset(dataset, batch_size):
    
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size, drop_remainder = True)
    # dataset = dataset.batch(tf.data.experimental.cardinality(dataset).numpy())
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    # dataset = np.array(list(dataset)[0]['image'])
    
    return dataset

def create_data_split(args):

    train_str = 'train[:{}%]'.format(args.percent_data_to_use)
    test_str = 'test[:{}%]'.format(args.percent_data_to_use)

    # train_str = 'small1[:{}%]'.format(args.percent_data_to_use)
    # test_str = 'small2[:{}%]'.format(args.percent_data_to_use)

    (full_train_set, test_dataset), ds_info = tfds.load('Omniglot', split = [train_str, test_str], shuffle_files = True, as_supervised = False, with_info = True)
    
    full_train_set = full_train_set.map(prepare_image, num_parallel_calls = tf.data.AUTOTUNE)

    n_data = tf.data.experimental.cardinality(full_train_set).numpy()
    n_data_validate = tf.cast(n_data * (args.fraction_for_validation), tf.int64)
    n_data_train = tf.cast(n_data * (1 - args.fraction_for_validation), tf.int64)
    
    train_dataset = full_train_set.take(n_data_train)
    validate_dataset = full_train_set.skip(n_data_train).take(n_data_validate)
    
    train_dataset = transform_train_dataset(train_dataset, args.batch_size_train, args.data_seed)
    validate_dataset = transform_validate_dataset(validate_dataset, args.batch_size_validate)
    #validate_dataset = np.array(next(iter(tfds.as_numpy(train_dataset)))['image'])
    
    return train_dataset, validate_dataset, test_dataset, ds_info
