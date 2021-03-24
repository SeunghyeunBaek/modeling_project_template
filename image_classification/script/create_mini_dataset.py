"""미니 데이터셋 생성
    1. 미니 데이터셋 생성
    2. Train, Validation, Test 분할
"""

import os
from module.util import sample_dataset, split_dataset

ORIGINAL_DATA_DIR = '/workspace/template_project/image_classification/data/original/'
SAMPLE_DATA_DIR = '/workspace/template_project/image_classification/data/sample10/'
SAMPLE_SPLIT_DATA_DIR = '/workspace/template_project/image_classification/data/sample10_splitted/'
SAMPLE_RATIO = 0.1
RANDOM_SEED = 42

if __name__ == '__main__':
    sample_dataset(data_dir=ORIGINAL_DATA_DIR, sample_data_dir=SAMPLE_DATA_DIR, sample_ratio=SAMPLE_RATIO)
    split_dataset(original_data_dir=SAMPLE_DATA_DIR,
                  splitted_data_dir=SAMPLE_SPLIT_DATA_DIR,
                  train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1,
                  is_stratify=True, random_seed=RANDOM_SEED, logger=None)

