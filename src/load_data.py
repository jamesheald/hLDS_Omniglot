import tensorflow as tf
import tensorflow_datasets as tfds
# hide any GPUs from TensorFlow, otherwise TF might reserve memory and make it unavailable to JAX
tf.config.set_visible_devices([], device_type = 'GPU')

# https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html
# https://www.tensorflow.org/datasets/catalog/omniglot
# https://www.tensorflow.org/datasets/api_docs/python/tfds/load

def prepare_image(dictionary):
    
    # remove redundant channels
    dictionary['image'] = dictionary['image'][:,:,0]
    
    # invert image so drawn pixels are 1
    dictionary['image'] = tf.cast(dictionary['image'] == 0, tf.float32)

    return dictionary

def transform_dataset(dataset, batch_size, tfds_seed):
    
    dataset = dataset.cache()
    dataset = dataset.shuffle(tf.data.experimental.cardinality(dataset).numpy(), seed = tfds_seed, reshuffle_each_iteration = True)
    dataset = dataset.batch(batch_size, drop_remainder = True)
    
    return dataset

def create_data_splits(cfg):

    (full_train_set, test_dataset), ds_info = \
    tfds.load('Omniglot', split = ['train', 'test'], shuffle_files = True, as_supervised = False, with_info = True)

    full_train_set = full_train_set.map(prepare_image, num_parallel_calls = tf.data.AUTOTUNE)

    num_data = tf.data.experimental.cardinality(full_train_set).numpy()
    train_dataset = full_train_set.take(num_data * (1 - cfg.validation_split))
    validate_dataset = full_train_set.take(num_data * (cfg.validation_split))
    train_dataset = transform_dataset(train_dataset, cfg.batch_size, cfg.tfds_seed)

    return train_dataset, validate_dataset, test_dataset, ds_info