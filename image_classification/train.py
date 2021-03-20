"""

"""


from module.dataloader import ImageDataset
from torch.utils.data import DataLoader

from module.trainer import BatchTrainer, PerformanceRecorder
from module.util import load_yaml, save_yaml, get_logger
from module.earlystoppers import LossEarlyStopper
from module.metrics import get_metric_function
from module.losses import get_loss_function
from module.optimizer import get_optimizer
from model.model import get_model, DNN

from datetime import datetime
import numpy as np
import random
import os

import torch

# CONFIG
PROJECT_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/config.yml')
SYSTEM_LOGGER_PATH = os.path.join(PROJECT_DIR, 'log/train.log')
config = load_yaml(CONFIG_PATH)

# DIRECTORY
DATA_DIR = config['DIRECTORY']['original_splitted_data']
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
VALIDATION_DATA_DIR = os.path.join(DATA_DIR, 'validation')
PERFORMANCE_RECORD_DIR = config['DIRECTORY']['performance_record']

# DATALOADER
NUM_WORKERS = config['DATALOADER']['num_workers']
SHUFFLE = config['DATALOADER']['shuffle']
PIN_MEMORY = config['DATALOADER']['pin_memory']
DROP_LAST = config['DATALOADER']['drop_last']

# TRAIN
MODEL_STR = 'DNN'
N_INPUT = config['TRAIN']['n_input']
N_OUTPUT = config['TRAIN']['n_output']
OPTIMIZER_STR = config['TRAIN']['optimizer']
LOSS_FUNCTION_STR = config['TRAIN']['loss_function']
METRIC_FUNCTION_STR = config['TRAIN']['metric_function']
EARLY_STOPPING_PATIENCE = config['TRAIN']['early_stopping_patience']

BATCH_SIZE = config['TRAIN']['batch_size']
EPOCH = config['TRAIN']['epoch']
LEARNING_RATE = config['TRAIN']['learning_rate']
MOMENTUM = config['TRAIN']['momentum']

# SEED
RANDOM_SEED = config['SEED']['random_seed']

# EXPERIMENT SERIAL: {model_name}_{timestamp}
EXPERIMENT_START_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_SERIAL = MODEL_STR + '_' + EXPERIMENT_START_TIMESTAMP

# PERFORMANCE_RECORD
PERFORMANCE_RECORD_COLUMN_NAME_LIST = config['PERFORMANCE_RECORD']['column_list']

if __name__ == '__main__':

    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set system logger
    system_logger = get_logger(name='train', file_path=SYSTEM_LOGGER_PATH)

    """
    Load data
    """

    train_dataset = ImageDataset(image_dir=os.path.join(TRAIN_DATA_DIR, 'image/'),
                                 label_path=os.path.join(TRAIN_DATA_DIR, 'label.json'))
    validation_dataset = ImageDataset(image_dir=os.path.join(VALIDATION_DATA_DIR, 'image/'),
                                      label_path=os.path.join(VALIDATION_DATA_DIR, 'label.json'))

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  num_workers=NUM_WORKERS, 
                                  shuffle=SHUFFLE,
                                  pin_memory=PIN_MEMORY)
    validation_dataloader = DataLoader(dataset=validation_dataset,
                                        batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORKERS, 
                                        shuffle=SHUFFLE,
                                        pin_memory=PIN_MEMORY)


    # Load model architecture
    model = get_model(model_str=MODEL_STR)
    model = model(n_input=N_INPUT, n_output=N_OUTPUT).to(device)

    # Load train module
    optimizer = get_optimizer(optimizer_str=OPTIMIZER_STR)
    optimizer = optimizer(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    loss_function = get_loss_function(loss_function_str=LOSS_FUNCTION_STR)
    metric_function = get_metric_function(metric_function_str=METRIC_FUNCTION_STR)

    # Batch trainer
    trainer = BatchTrainer(model=model,
                           optimizer=optimizer,
                           loss_function=loss_function,
                           metric_function=metric_function,
                           device=device,
                           logger=system_logger)

    # Performance recorder
    performance_recorder = PerformanceRecorder(serial=EXPERIMENT_SERIAL,
                                               column_list=PERFORMANCE_RECORD_COLUMN_NAME_LIST,
                                               root_dir=PERFORMANCE_RECORD_DIR)

    # Early stopper
    early_stopper = LossEarlyStopper(patience=EARLY_STOPPING_PATIENCE,
                                    weight_path=os.path.join(performance_recorder.record_dir, 'model.pth'),
                                    logger=system_logger,
                                    verbose=True)
    # Performance recorder key row
    key_row_list = [EXPERIMENT_SERIAL,
                    EXPERIMENT_START_TIMESTAMP,
                    MODEL_STR, OPTIMIZER_STR,
                    LOSS_FUNCTION_STR,
                    METRIC_FUNCTION_STR,
                    EARLY_STOPPING_PATIENCE,
                    BATCH_SIZE, EPOCH,
                    LEARNING_RATE,
                    MOMENTUM,
                    RANDOM_SEED]                    

    # Train
    for epoch_index in range(EPOCH):
        trainer.train_batch(dataloader=train_dataloader, epoch_index=epoch_index, verbose=False)
        trainer.validate_batch(dataloader=validation_dataloader, epoch_index=epoch_index, verbose=False)
        early_stopper.check_early_stopping(loss=trainer.validation_loss_mean, model=trainer.model)
                
        # Performance record
        epoch_row_list = key_row_list + [epoch_index,
                                         trainer.train_loss_mean,
                                         trainer.validation_loss_mean,
                                         trainer.train_score,
                                         trainer.validation_score,
                                         early_stopper.stop]
        performance_recorder.add_row(epoch_row_list)

        # Clear epoch history
        trainer.clear_history()

        # Early stopping
        if early_stopper.stop:
            break
        
    # Save config
    save_yaml(os.path.join(performance_recorder.record_dir, 'config.yml'), config)