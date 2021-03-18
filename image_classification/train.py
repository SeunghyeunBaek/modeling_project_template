"""

"""


from module.dataloader import ImageDataset
from torch.utils.data import DataLoader

from module.util import load_yaml, set_seed, get_logger
from module.earlystoppers import LossEarlyStopper
from module.metrics import get_metric_function
from module.losses import get_loss_function
from module.optimizer import get_optimizer
from module.trainer import BatchTrainer
from model.dnn import DNN
import torch
import os

# CONFIG
PROJECT_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/config.yml')
SYSTEM_LOGGER_PATH = os.path.join(PROJECT_DIR, 'log/train.log')
config = load_yaml(CONFIG_PATH)

# DIRECTORY
DATA_DIR = config['DIRECTORY']['original_splitted_data']
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
VALIDATION_DATA_DIR = os.path.join(DATA_DIR, 'validation')

# DATALOADER
NUM_WORKERS = config['DATALOADER']['num_workers']
SHUFFLE = config['DATALOADER']['shuffle']
PIN_MEMORY = config['DATALOADER']['pin_memory']
DROP_LAST = config['DATALOADER']['drop_last']

# TRAIN
BATCH_SIZE = config['TRAIN']['batch_size']
EPOCH = config['TRAIN']['epoch']
LEARNING_RATE = config['TRAIN']['learning_rate']
MOMENTUM = config['TRAIN']['momentum']
OPTIMIZER_STR = config['TRAIN']['optimizer']
LOSS_FUNCTION_STR = config['TRAIN']['loss_function']
METRIC_FUNCTION_STR = config['TRAIN']['metric_function']
EARLY_STOPPING_PATIENCE = config['TRAIN']['early_stopping_patience']


# SEED
RANDOM_SEED = config['SEED']['random_seed']


if __name__ == '__main__':

    set_seed(RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = get_logger(name='train', file_path=SYSTEM_LOGGER_PATH)

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


    model = DNN().to(device)
    optimizer = get_optimizer(optimizer_str=OPTIMIZER_STR)
    optimizer = optimizer(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    loss_function = get_loss_function(loss_function_str=LOSS_FUNCTION_STR)
    metric_function = get_metric_function(metric_function_str=METRIC_FUNCTION_STR)
    early_stopper = LossEarlyStopper(patience=EARLY_STOPPING_PATIENCE, weight_path='model.pth', verbose=True)

    trainer = BatchTrainer(model=model,
                           optimizer=optimizer,
                           loss_function=loss_function,
                           metric_function=metric_function,
                           device=device,
                           logger=logger)

    for epoch_index in range(EPOCH):
        trainer.train_batch(dataloader=train_dataloader, epoch_index=epoch_index, verbose=False)
        trainer.validate_batch(dataloader=validation_dataloader, epoch_index=epoch_index, verbose=False)
        early_stopper.check_early_stopping(loss=trainer.validation_loss_mean, model=trainer.model)
        
        if early_stopper.stop:
            break