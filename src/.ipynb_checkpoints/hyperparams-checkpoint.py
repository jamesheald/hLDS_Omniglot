from ml_collections import config_dict

def get_hyperparameter_configuration():
  
  cfg = config_dict.ConfigDict()
  cfg.learning_rate = 0.1
  cfg.momentum = 0.9
  cfg.batch_size = 128
  cfg.num_epochs = 10
  cfg.opt = 10

  return cfg