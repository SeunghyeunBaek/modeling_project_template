"""
데이터셋 분할
"""

import os
import sys
from glob import glob
import shutil
import csv

from numpy.random.mtrand import shuffle
from torch.utils.data.dataset import random_split

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_DIR)

from modules.utils import make_directory, load_yaml, save_yaml, load_json, save_json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

script_name = os.path.basename(__file__)
config_name = script_name.replace('.py', '_config.yml')
config_path = os.path.join(PROJECT_DIR, 'config', 'scripts', config_name)
config = load_yaml(config_path)

SRC_DATA_DIR = os.path.join(PROJECT_DIR, 'data', config['DATASET']['src_dataset'])
DST_DATA_DIR = os.path.join(PROJECT_DIR, 'data', config['DATASET']['dst_dataset'])

if __name__ == '__main__':

    src_img_dir = os.path.join(SRC_DATA_DIR, 'images')
    src_label_path = os.path.join(SRC_DATA_DIR, 'label.json')
    src_label = load_json(src_label_path)

    src_filenames = list(src_label.keys())
    src_targets = list(src_label.values())

    # Train / val + test
    x_train, x_valtest, y_train, y_valtest = train_test_split(src_filenames, 
                                                             src_targets,
                                                             test_size=config['SPLIT']['val']+config['SPLIT']['test'],
                                                             shuffle=True,
                                                             stratify=src_targets,
                                                             random_state=config['SPLIT']['seed'])
    
    # val / test
    test_val_rate = config['SPLIT']['test']/(config['SPLIT']['val'] + config['SPLIT']['test'])
    x_val, x_test, y_val, y_test = train_test_split(x_valtest,
                                                    y_valtest,
                                                    test_size=test_val_rate,
                                                    shuffle=True,
                                                    stratify=y_valtest,
                                                    random_state=config['SPLIT']['seed'])
    
    train_label = dict(zip(x_train, y_train))
    val_label = dict(zip(x_val, y_val))
    test_label = dict(zip(x_test, y_test))

    phase_dict = {
        'train': train_label,
        'val': val_label,
        'test': test_label
    }

    for phase, label in phase_dict.items():

        print(f'Split dataset {phase}')

        dst_image_dir = os.path.join(DST_DATA_DIR, phase, 'images')
        make_directory(dst_image_dir)
        
        for filename, target in tqdm(list(label.items())):

            # Save image
            src_image_path = os.path.join(SRC_DATA_DIR, 'images', filename)
            shutil.copy(src_image_path, dst_image_dir)

            # Record filename
            with open(os.path.join(DST_DATA_DIR, 'filenames.csv'), 'a') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'target', 'phase'])
                row = {'filename': filename, 'target': target, 'phase': phase}
                
                if f.tell() == 0:
                    writer.writeheader()

                writer.writerow(row)

        # Save label
        save_json(os.path.join(DST_DATA_DIR, phase, 'label.json'), label)
    
    # Save config
    save_yaml(os.path.join(DST_DATA_DIR, config_name), config)

