"""Predict

"""
from modules.utils import load_yaml, save_pickle, save_yaml, get_logger, make_directory, save_json

from modules.earlystoppers import LossEarlyStopper
from modules.recorders import Recorder
from modules.optimizers import get_optimizer
from modules.datasets import ImageDataset
from modules.trainer import Trainer

from modules.preprocessor import get_preprocessor
from modules.metrics import get_metric
from modules.losses import get_loss
from models.utils import get_model

from torch.utils.data import DataLoader

from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import numpy as np
import random
import os
import copy

from apex import amp
import torch

# Config
PROJECT_DIR = os.path.dirname(__file__)
predict_config = load_yaml(os.path.join(PROJECT_DIR, 'config/predict_config.yml'))

# Serial
train_serial = predict_config['TRAIN']['train_serial']
kst = timezone(timedelta(hours=9))
predict_timestamp = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")
predict_serial = train_serial + '_' + predict_timestamp

# Predict directory
PREDICT_DIR = os.path.join(PROJECT_DIR, 'results', 'predict', predict_serial)
make_directory(PREDICT_DIR)

# Data Directory
DATA_DIR = os.path.join(PROJECT_DIR,
                         'data',
                          predict_config['DIRECTORY']['dataset'],
                          predict_config['DIRECTORY']['phase'])

# Train config
RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', train_serial)
train_config = load_yaml(os.path.join(RECORDER_DIR, 'train_config.yml'))

# SEED
torch.manual_seed(predict_config['PREDICT']['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(predict_config['PREDICT']['seed'])
random.seed(predict_config['PREDICT']['seed'])

# Gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(predict_config['PREDICT']['gpu'])

if __name__ == '__main__':

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    preprocessors = [get_preprocessor(preprocessor) for preprocessor in train_config['PREPROCESS']]
    test_dataset = ImageDataset(image_dir=os.path.join(DATA_DIR, 'images'),
                                label_path=os.path.join(DATA_DIR, 'label.json'),
                                preprocessors=preprocessors)
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=train_config['DATALOADER']['batch_size'],
                                num_workers=train_config['DATALOADER']['num_workers'], 
                                shuffle=False,
                                pin_memory=train_config['DATALOADER']['pin_memory'],
                                drop_last=train_config['DATALOADER']['drop_last'])

    # Load model
    model_name = train_config['TRAINER']['model']
    model_args = train_config['MODEL'][model_name]
    model = get_model(model_name=model_name, model_args=model_args).to(device)

    checkpoint = torch.load(os.path.join(RECORDER_DIR, 'model.pt'))
    
    #!AMP
    if train_config['TRAINER']['amp'] == True:
        model = amp.initialize(model, opt_level='O1')
        model.load_state_dict(checkpoint['model'])
        amp.load_state_dict(checkpoint['amp'])
    else:
        model.load_state_dict(checkpoint['model'])

    model.eval()

    gt_dict, pred_dict = dict(), dict()

    y_logits_dir = os.path.join(PREDICT_DIR, 'y_logits')
    make_directory(y_logits_dir)

    for batch_index, (x, y, filename) in enumerate(tqdm(test_dataloader)):
        x = x.to(device)
        y_logits = model(x).cpu()
        y_pred = torch.argmax(y_logits, dim=1)
        y_logits = y_logits.detach().numpy()
        y_pred = y_pred.detach().numpy()
        y = y.numpy()

        for y_, filename_, y_logit, y_pred_ in zip(y, filename, y_logits, y_pred):
            gt_dict[filename_] = int(y_)
            pred_dict[filename_] = int(y_pred_)
            save_pickle(os.path.join(y_logits_dir, filename_+'.pkl'), y_logit)

    save_json(os.path.join(PREDICT_DIR, 'gt.json'), gt_dict)
    save_json(os.path.join(PREDICT_DIR, 'pred.json'), pred_dict)