from model.model import get_model, DNN
from module.util import load_yaml
from datatime import datetime
import torch
import os

# CONFIG
PROJECT_DIR = os.path.dirname(__file__)
PREDICT_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/perdict_config.yml')
SYSTEM_LOGGER_PATH = os.path.join(PROJECT_DIR, 'log/predict.log')
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
MODEL_PATH = os.path.join(TRAIN_RESULT_DIR, 'model.pth')
TRAIN_CONFIG_DIR = os.path.join(TRAIN_RESULT_DIR, 'config.yml')

# DATALOADER
train_config = load_yaml(TRAIN_CONFIG_DIR)
NUM_WORKERS = train_config['DATALOADER']['num_workers']
SHUFFLE = train_config['DATALOADER']['shuffle']
PIN_MEMORY = train_config['DATALOADER']['pin_memory']
DROP_LAST = train_config['DATALOADER']['drop_last']

# MODEL
MODEL_STR = train_config['TRAIN']['model']
BATCH_SIZE = train_config['TRAIN']['batch_size']
METRIC_FUNCTION_STR = train_config['TRAIN']['metric_function']
LOSS_FUNCTION_STR = train_config['TRAIN']['loss_function']
N_INPUT = train_config['TRAIN']['n_input']
N_OUTPUT = train_config['TRAIN']['n_output']
# OPTIMIZER_STR = train_config['TRAIN']['OPTIMIZER']


if __name__ == '__main__':
    
    model = get_model(MODEL_STR)
    model = model(n_input=N_INPUT, n_output=N_OUTPUT)

    ## Load model
    model.load_state_dict(torch.load(MODEL_PATH))
    
    