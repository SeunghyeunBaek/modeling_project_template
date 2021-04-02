"""학습 스크립트

TODO:
    * plot_performance, PerformanceRecorder 로 병합 | Done
    * Epoch 단위 기록 PerformanceRecorder 로 병합 | Done
    * Best train recored 기록 스크립트 작성 | Done
    * Early stopping save model -> performance recorder | Done
    * Performance recorder result directory 생성 함수 train.py 이동 | Done
    * Docstring 작성
"""

from modules.utils import load_yaml, save_yaml, get_logger, make_directory
from modules.earlystoppers import LossEarlyStopper
from modules.recorders import PerformanceRecorder
from modules.metrics import get_metric_function
from modules.losses import get_loss_function
from modules.optimizers import get_optimizer
from modules.datasets import ImageDataset
from modules.trainer import BatchTrainer
from models.utils import get_model
from models.dnn import DNN

from torch.utils.data import DataLoader

from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import numpy as np
import random
import os

import torch

# CONFIG
PROJECT_DIR = os.path.dirname(__file__)
TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/train_config.yml')
SYSTEM_LOGGER_PATH = os.path.join(PROJECT_DIR, 'log/train.log')
config = load_yaml(TRAIN_CONFIG_PATH)

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
MODEL_STR = config['TRAIN']['model']
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

# TRAIN SERIAL: {model_name}_{timestamp}
# time offset set
KST = timezone(timedelta(hours=9))
TRAIN_START_TIMESTAMP = datetime.now(tz=KST).strftime("%Y%m%d_%H%M%S")
TRAIN_SERIAL = MODEL_STR + '_' + TRAIN_START_TIMESTAMP

# PERFORMANCE_RECORD
PERFORMANCE_RECORD_COLUMN_NAME_LIST = config['PERFORMANCE_RECORD']['column_list']
PERFORMANCE_RECORD_DIR = os.path.join(config['DIRECTORY']['performance_record'], TRAIN_SERIAL)

if __name__ == '__main__':

    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set train result directory
    make_directory(PERFORMANCE_RECORD_DIR)

    # Set system logger: 로거 객체
    system_logger = get_logger(name='train',
                               file_path=os.path.join(PERFORMANCE_RECORD_DIR, 'train_log.log'))

    # Load dataset
    train_dataset = ImageDataset(image_dir=os.path.join(TRAIN_DATA_DIR, 'image/'),
                                 label_path=os.path.join(TRAIN_DATA_DIR, 'label.json'))
    validation_dataset = ImageDataset(image_dir=os.path.join(VALIDATION_DATA_DIR, 'image/'),
                                      label_path=os.path.join(VALIDATION_DATA_DIR, 'label.json'))

    # Load dataloader
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
    # Load model
    model = get_model(model_str=MODEL_STR)
    model = model(n_input=N_INPUT, n_output=N_OUTPUT).to(device)

    # Load optimizerm loss function, metric function
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

    # Early stopper
    early_stopper = LossEarlyStopper(patience=EARLY_STOPPING_PATIENCE,
                                     logger=system_logger,
                                     verbose=True)

    # Set performance_recorder: 성능을 기록하는 객체
    key_column_value_list = [
        TRAIN_SERIAL,
        TRAIN_START_TIMESTAMP,
        MODEL_STR,
        OPTIMIZER_STR,
        LOSS_FUNCTION_STR,
        METRIC_FUNCTION_STR,
        EARLY_STOPPING_PATIENCE,
        BATCH_SIZE,
        EPOCH,
        LEARNING_RATE,
        MOMENTUM,
        RANDOM_SEED]  

    performance_recorder = PerformanceRecorder(column_name_list=PERFORMANCE_RECORD_COLUMN_NAME_LIST,
                                               record_dir=PERFORMANCE_RECORD_DIR,
                                               key_column_value_list=key_column_value_list,
                                               logger=system_logger,
                                               model=model)   
    # Train
    for epoch_index in tqdm(range(EPOCH)):
        trainer.train_batch(dataloader=train_dataloader, epoch_index=epoch_index, verbose=False)
        trainer.validate_batch(dataloader=validation_dataloader, epoch_index=epoch_index, verbose=False)

        early_stopper.check_early_stopping(loss=trainer.validation_loss_mean)
                
        # Performance record - csv
        performance_recorder.add_row(epoch_index=epoch_index,
                                     train_loss=trainer.train_loss_mean,
                                     validation_loss=trainer.validation_loss_mean,
                                     train_score=trainer.train_score,
                                     validation_score=trainer.validation_score)
        
        # Performance record - plot
        performance_recorder.save_performance_plot(final_epoch=epoch_index)

        # Clear trainer epoch history
        trainer.clear_history()
        
        # Early stopping
        if early_stopper.stop:
            break
    
    # Save best performance
    performance_recorder.add_best_row()

    # Save config
    save_yaml(os.path.join(performance_recorder.record_dir, 'train_config.yml'), config)