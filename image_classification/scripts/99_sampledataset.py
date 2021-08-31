"""
데이터셋 샘플링
"""

import os
import sys
from glob import glob
import shutil
import random

from numpy.random.mtrand import shuffle
from torch.serialization import save
from torch.utils.data.dataset import random_split

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_DIR)

from modules.utils import make_directory, load_yaml, save_yaml, load_json, save_json
from tqdm import tqdm

script_name = os.path.basename(__file__)
config_name = script_name.replace('.py', '_config.yml')
config_path = os.path.join(PROJECT_DIR, 'config', 'scripts', config_name)
config = load_yaml(config_path)

SRC_DATA_DIR = os.path.join(PROJECT_DIR, 'data', config['DATASET']['src_dataset'])
DST_DATA_DIR = os.path.join(PROJECT_DIR, 'data', config['DATASET']['dst_dataset'])

if __name__ == '__main__':

    random.seed(config['SAMPLE']['seed'])

    phases = ['train', 'val', 'test']

    for phase in phases:
        
        print(f'Sample dataset, {phase}')

        src_image_dir = os.path.join(SRC_DATA_DIR, phase, 'images')
        src_label_path = os.path.join(SRC_DATA_DIR, phase, 'label.json')
        src_label = load_json(src_label_path)

        src_filenames = list(src_label.keys())
        src_targets = list(src_label.values())

        n_samples = round(config['SAMPLE']['ratio']*len(src_filenames))
        sampled_filenames = random.sample(src_filenames, n_samples)

        dst_label = dict()
        dst_image_dir = os.path.join(DST_DATA_DIR, phase, 'images')
        make_directory(dst_image_dir)
        
        for sampled_filename in tqdm(sampled_filenames):

            src_image_path = os.path.join(src_image_dir, sampled_filename)
            target = src_label[sampled_filename]

            shutil.copy(src_image_path, dst_image_dir)
            dst_label[sampled_filename] = target

        save_json(os.path.join(DST_DATA_DIR, phase, 'label.json'), dst_label)

    save_yaml(os.path.join(DST_DATA_DIR, config_name), config)
    



