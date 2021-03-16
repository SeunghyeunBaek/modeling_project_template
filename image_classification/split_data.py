"""데이터 분할

    * 데이터를 train, test, validation 으로 분할
    * config.yaml 에서 분할 비율 설정
    * 새로운 경로에 분할된 데이터 저장

TOOD:
    * 
"""

from module.util import split_dataset, load_yaml, get_logger
import os

if __name__ == '__main__':
    
    # Config 불러오기
    PROJECT_DIR = os.path.dirname(__file__)
    CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/config.yml')
    config_dict = load_yaml(CONFIG_PATH)

    # 경로
    ORIGINAL_DATA_DIR = config_dict['DIRECTORY']['raw']
    SPLITTED_DATA_DIR = config_dict['DIRECTORY']['raw_splitted']
    LOG_DIR = config_dict['DIRECTORY']['system_log']

    # 분할 설정
    TRAIN_RATIO = config_dict['SPLIT_DATA']['train_ratio']
    VALIDATION_RATIO = config_dict['SPLIT_DATA']['validation_ratio']
    TEST_RATIO = config_dict['SPLIT_DATA']['test_ratio']

    RANDOME_SEED = config_dict['SEED']['random_seed']
    IS_STRATIFY = config_dict['SPLIT_DATA']['is_stratify']
    
    # Logger 설정
    logger = get_logger(name='split_data', file_path=os.path.join(LOG_DIR, 'split_data.log'))
    
    logger.info(f"Start split dataset")
    logger.info(f"Set original data directory: {ORIGINAL_DATA_DIR}")
    logger.info(f"Set splitted data directory: {SPLITTED_DATA_DIR}")
    logger.info(f"Set train:validation:test={TRAIN_RATIO}:{VALIDATION_RATIO}:{TEST_RATIO} stratify: {IS_STRATIFY} seed: {RANDOME_SEED}")

    split_dataset(original_data_dir=ORIGINAL_DATA_DIR,
                  splitted_data_dir=SPLITTED_DATA_DIR,
                  train_ratio=TRAIN_RATIO,
                  validation_ratio=VALIDATION_RATIO,
                  test_ratio=TEST_RATIO,
                  is_stratify=IS_STRATIFY,
                  random_seed=RANDOME_SEED,
                  logger=logger)