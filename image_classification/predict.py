"""Predict

TODO:
    Docstring 작성
"""
from module.util import load_yaml, get_logger, save_yaml, make_directory, get_tpfp_mapper
from module.metrics import get_metric_function
from module.losses import get_loss_function
from module.dataloader import ImageDataset
from module.trainer import BatchTrainer
from model.model import get_model, DNN


from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np
import random
import torch
import os

# CONFIG
PROJECT_DIR = os.path.dirname(__file__)
PREDICT_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/predict_config.yml')
predict_config = load_yaml(PREDICT_CONFIG_PATH)

# SERIAL
EXPERIMENT_SERIAL = predict_config['EXPERIMENT']['serial']
PREDICT_START_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
PREDICT_SERIAL = EXPERIMENT_SERIAL + '_' + PREDICT_START_TIMESTAMP

# DIRECTORY
TEST_DATA_DIR = predict_config['DIRECTORY']['data']
RESULT_DIR = predict_config['DIRECTORY']['result']
TRAIN_RESULT_DIR = os.path.join(os.path.join(RESULT_DIR, 'train'), EXPERIMENT_SERIAL)
PREDICT_RESULT_DIR = os.path.join(os.path.join(RESULT_DIR, 'predict'), PREDICT_SERIAL)

TRAIN_CONFIG_DIR = os.path.join(TRAIN_RESULT_DIR, 'config.yml')
MODEL_PATH = os.path.join(TRAIN_RESULT_DIR, 'model.pth')

# DATALOADER
train_config = load_yaml(TRAIN_CONFIG_DIR)
NUM_WORKERS = train_config['DATALOADER']['num_workers']
PIN_MEMORY = train_config['DATALOADER']['pin_memory']
DROP_LAST = train_config['DATALOADER']['drop_last']

# MODEL
MODEL_STR = train_config['TRAIN']['model']
BATCH_SIZE = train_config['TRAIN']['batch_size']
METRIC_FUNCTION_STR = train_config['TRAIN']['metric_function']
LOSS_FUNCTION_STR = train_config['TRAIN']['loss_function']
N_INPUT = train_config['TRAIN']['n_input']
N_OUTPUT = train_config['TRAIN']['n_output']

# SEED
RANDOM_SEED = train_config['SEED']['random_seed']

if __name__ == '__main__':

    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    test_dataset = ImageDataset(image_dir=os.path.join(TEST_DATA_DIR, 'image/'),
                                label_path=os.path.join(TEST_DATA_DIR, 'label.json'))
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCH_SIZE,
                                 num_workers=NUM_WORKERS,
                                 shuffle=False,
                                 pin_memory=PIN_MEMORY)

    # Load model
    model = get_model(MODEL_STR)
    model = model(n_input=N_INPUT, n_output=N_OUTPUT).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))

    loss_function = get_loss_function(loss_function_str=LOSS_FUNCTION_STR)
    metric_function = get_metric_function(metric_function_str=METRIC_FUNCTION_STR)

    # Set trainer
    trainer = BatchTrainer(model=model,
                           loss_function=loss_function,
                           metric_function=metric_function,
                           device=device)

    # Test
    trainer.validate_batch(dataloader=test_dataloader, verbose=False)
    
    try:
        # Save test result
        make_directory(PREDICT_RESULT_DIR)
        predict_result_dict = {
            'test_target_list': trainer.validation_target_list,
            'test_target_pred_list': trainer.validation_target_pred_list,
            'test_filename_list': trainer.validation_image_filename_list,
            'test_loss': float(trainer.validation_loss_mean),
            'test_score': float(trainer.validation_score),
        }

        save_yaml(os.path.join(PREDICT_RESULT_DIR, 'record.yml'), predict_result_dict)
        save_yaml(os.path.join(PREDICT_RESULT_DIR, 'train_config.yml'), train_config)
        save_yaml(os.path.join(PREDICT_RESULT_DIR, 'predict_config.yml'), predict_config)

    except Exception as e:
        print(f"Error {e}")