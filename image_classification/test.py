import os
import shutil
from glob import glob

PRJ_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(PRJ_DIR, 'data/')

if __name__ == '__main__':

    train_path_list = glob(DATA_DIR + '*.png', recursive=True)
    test_path_list = glob(DATA_DIR + '*.png', recursive=True)
    
    print(train_path_list)
    pass