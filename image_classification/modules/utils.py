"""공용 함수
    
    * File IO
    * Logger
    * System
    * Seed
    * Plotter
TODO:
    * docstring 작성
    * get_tpfp_mapper 이름 바꾸기

Notes:
    * deprecated split_dataset
        - 경로 구조 변경됨

Reference:    
    * set_seed: https://hoya012.github.io/blog/reproducible_pytorch/
"""

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from itertools import product
import numpy as np
import logging
import random
import pickle
import shutil
import torch
import json
import yaml
import csv
import os

"""
File IO
"""
def save_pickle(path, obj):
    
    with open(path, 'wb') as f:
        
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(path):

    with open(path, 'rb') as f:

        return pickle.load(f)


def save_json(path, obj, sort_keys=True)-> str:
    
    try:
        
        with open(path, 'w') as f:
            
            json.dump(obj, f, indent=4, sort_keys=sort_keys)
        
        msg = f"Json saved {path}"
    
    except Exception as e:
        msg = f"Fail to save {e}"

    # print(msg)
    return msg

def load_json(path):

	with open(path, 'r', encoding='utf-8') as f:

		return json.load(f)


def save_yaml(path, obj):
	
	with open(path, 'w') as f:

		yaml.dump(obj, f, sort_keys=False)
		

def load_yaml(path):

	with open(path, 'r') as f:
		return yaml.load(f, Loader=yaml.FullLoader)


# deprecated
# def load_dir_list(path):
    
#     return os.listdir(path)    


"""
Logger
"""
def get_logger(name: str, dir_: str, stream=False)-> logging.RootLogger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # logging all levels
    
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(dir_, f'{name}.log'))

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if stream:
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

"""
System
"""
def select_copy_file(from_directory: str,
                     to_directory: str,
                     filename_list: list)-> str:
    """파일 선택 후 복사

    filename_list 에 해당하는 파일을 to_directory 로 복사

    Args:
        from_directory (str): 원 경로
        to_directory (str): 복사 경로
        filename_list (list(str)): 파일명 리스트

    Returns:
        str: 상태 메시지

    """

    filepath_list = [os.path.join(from_directory, filename) for filename in filename_list]

    for filepath in filepath_list:
        filename = filepath.split('/')[-1]
        to_filepath = os.path.join(to_directory, filename)
        shutil.copy(filepath, to_filepath)
    
    msg = f"{len(filepath_list)} files copied to {to_directory}"
    print(msg)

    return msg


def make_directory(directory: str)-> str:
    """경로가 없으면 생성
    
    Args:
        directory (str): 새로 만들 경로

    Returns:
        str: 상태 메시지

    """

    try:
        
        if not os.path.isdir(directory):
            os.makedirs(directory)
            msg = f"Create directory {directory}"
        
        else:
            msg = f"{directory} already exists"

    except OSError as e:
        msg = f"Fail to create directory {directory} {e}"
        # print(e)
    return msg


def sample_dataset(data_dir:str, sample_data_dir: str, sample_ratio:float):
    """데이터 샘플링
        
        원본 데이터 경로와 동일한 디렉토리 구조 생성 후 저장
        
    Args:
        data_dir (str): 원본 데이터 경로
        sample_data_dir (str): 샘플링 데이터 저장 경로
        sample_ratio (float): 샘플 비율
        
    Note:
        data_dir 은 아래와 같은 구성이라고 가정
        ```
        data_dir/
            \_image/
                \_image001.png
                \_image002.png
                ...
            \_label.json/
        ```
    """

    # 경로 생성
    sample_image_dir = os.path.join(sample_data_dir, 'image')
    sample_label_path = os.path.join(sample_data_dir, 'label.json')
    msg = make_directory(sample_image_dir) + '\n'

    # 라벨 파일 불러오기
    all_label_dict = load_json(os.path.join(data_dir, 'label.json'))
    all_filename_list = list(all_label_dict.keys())
    
    # 샘플링
    n_data = len(all_filename_list)
    n_sample = round(n_data*sample_ratio)
    sample_filename_list = random.sample(all_filename_list, n_sample)
    sample_label_dict = {filename: target for filename, target in all_label_dict.items() if filename in sample_filename_list}

    # 저장
    select_copy_file(from_directory=os.path.join(data_dir, 'image'),
                     to_directory=sample_image_dir,
                     filename_list=sample_filename_list)

    save_json(path=sample_label_path, obj=sample_label_dict)
    
    msg += f"Sampling {sample_ratio}, {n_data} -> {n_sample} saved {sample_data_dir}"
    print(msg)


# Deprecated(경로 구성 바뀜)
# def split_dataset(original_data_dir: str, splitted_data_dir: str, 
#                   train_ratio: float, validation_ratio: float, test_ratio: float, 
#                   is_stratify: bool,  random_seed: int, logger: logging.RootLogger=None)-> None:

#     """train,test,validation 데이터 셋 생성
        
#         original_data_dir 에서 일정 비율(config.yaml)로 train, validation, test 분리
#         splitted_data_dir 에 train, validation, test 경로 생성 후 이미지, 라벨 저장
#         경로 구성 방식은 Note 참고

#         Args:
#             original_data_dir (str): 원본 데이터 경로
#             splitted_data_dir (str): 분할 데이터 경로
#             train_ratio (float): Train 데이터 비율
#             validation_ratio (float): Validation 데이터 비율
#             test_ratio (float): Test 데이터 비율
#             is_stratify (bool): True 로 설정 시 label dictionary value를 target 으로 stratify split 진행
#             random_seed (int): random seed
#             logger (`logging.RootLogger`, optional): 미설정 시 print로 메시지 출력

#         Raises:
#             AssertionError: train_ratio + validation_ratio + test_ratio != 1

#         Note:
#             original_data_dir 은 아래와 같은 구성이라고 가정
#                 ```

#                 original_data_dir/
#                     \_image/
#                         \_image001.png
#                         \_image002.png
#                         ...
#                     \_label.json/

#                 ```

#             splitted_data_dir 은 아래와 같이 생성
#                 ```

#                 splitted_data_dir/
#                     \_train
#                         \_image/
#                         \_label.json
#                     \_validation
#                         \_image/
#                         \_label.json
#                     \_test
#                         \_image/
#                         \_label.json

#                 ```

#             label.json 의 key 는 파일명(확장자 포함) value 는 target 으로 구성
#                 ```label.json

#                 {'image001': 0,
#                  'image002': 1,
#                  ...}

#                 ```
#     """

#     # Train, validation, test 비율 합이 1이 아닐 경우 오류 발생
#     assert train_ratio + validation_ratio + test_ratio == 1,\
#            f'Dataset ratio error {train_ratio}:{validation_ratio}:{test_ratio}'

#     # 라벨 파일 불러오기
#     all_label_dict = load_json(os.path.join(original_data_dir, 'label.json'))
#     all_filename_list = list(all_label_dict.keys())
#     all_target_list = list(all_label_dict.values())

#     # test / validation 분할 비율 계산
#     test_validation_ratio = round(test_ratio / (validation_ratio + test_ratio), 1)
    
#     # 데이터셋 분할
#     train_filename_list, other_filename_list,\
#     train_target_list, other_target_list = train_test_split(all_filename_list,all_target_list,
#                                                             test_size=test_ratio,
#                                                             random_state=random_seed,
#                                                             stratify=all_target_list if is_stratify else None)

#     validation_filename_list, test_filename_list,\
#     validation_target_list, test_target_list = train_test_split(other_filename_list,other_target_list,
#                                                                 test_size=test_validation_ratio,
#                                                                 random_state=random_seed,
#                                                                 stratify=other_target_list if is_stratify else None)
    
#     # 경로 생성, 라벨, 데이터  저장
#     phase_list = ['train', 'validation', 'test']
#     phase_filename_list = [train_filename_list, validation_filename_list, test_filename_list]
#     phase_target_list = [train_target_list, validation_target_list, test_target_list]

#     # Train, validation, test 에 대해 각각 진행(경로 생성, 라벨 생성, 라벨 저장, 이미지 저장)
#     for phase, filename_list, target_list in zip(phase_list, phase_filename_list, phase_target_list):
#         start_msg = f"Start {phase} data copy"
#         logger.info(start_msg) if logger else print(start_msg)
        
#         new_directory = os.path.join(splitted_data_dir, phase+'/image')                            # 새 경로 지정

#         mkdir_msg = make_directory(new_directory)                                                  # 새 경로 생성
#         logger.info(mkdir_msg) if logger else print(mkdir_msg)

#         label_dict = dict(zip(filename_list, target_list))                                         # 라벨 생성
#         save_msg = save_json(os.path.join(splitted_data_dir, phase+'/label.json'), label_dict)     # 라벨 저장
#         logger.info(save_msg) if logger else print(save_msg)

#         copy_msg = select_copy_file(from_directory=os.path.join(original_data_dir, 'image/'),
#                                     to_directory=new_directory,
#                                     filename_list=filename_list)                                   # 이미지 저장
#         logger.info(copy_msg) if logger else print(copy_msg)


#TODO: 이름 변경 필요
def get_tpfp_mapper(y_list: list, y_pred_list: list, filename_list: list)-> dict:
    """
    클래스별 tp fp 와 filename 매핑

    Args:
        y_list (list):
        y_pred_list (list):
        filename_list (list):

    Returns
        dict
    """

    tpfp_mapper_dict = dict()
    unique_y_list = list(set(y_pred_list))

    # Initiate for all key
    for key in product(unique_y_list, repeat=2):
        tpfp_mapper_dict[key] = list()

    # Map true positive, false positive
    for y, y_pred, filename in zip(y_list, y_pred_list, filename_list):
        key = (y, y_pred)
        tpfp_mapper_dict[key].append(filename)

    return tpfp_mapper_dict


def count_csv_row(path):
    """
    CSV 열 수 세기
    """
    with open(path, 'r') as f:
        reader = csv.reader(f)
        n_row = sum(1 for row in reader)

    return n_row

if __name__ == '__main__':
    pass