"""evaluate

"""

from module.metrics import get_metrics_function
from module.util import load_yaml
import os


PREDICT_SERIAL = 'DNN_20210322_070350_20210322_083126'

RESULT_DIR = '/workspace/template_project/image_classification/result/'
PREDICT_DIR = os.path.join(os.path.join(RESULT_DIR, PREDICT_SERIAL))
PREDICT_RECORD_PATH = os.path.join(PREDICT_DIR, 'record.yml')
TRAIN_CONFIG_PATH = os.path.join(PREDICT_DIR, 'train_config.yml')


if __name__ == '__main__':
    pass

    
