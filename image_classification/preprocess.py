"""
전처리 한번에 실행
"""

from itertools import product
import subprocess
import sys
import os

PROJECT_DIR = os.path.dirname(__file__)
SCRIPT_DIR = os.path.join(PROJECT_DIR, 'scripts')   # 전처리 스크립트 경로
CONFIG_DIR = os.path.join(PROJECT_DIR, 'config')    # 전처리 설정 파일 경로

from modules.utils import load_yaml, save_yaml

script_filename = os.path.basename(os.path.abspath(__file__))
config_filename = script_filename.replace('.py', '_config.yml')
config = load_yaml(os.path.join(CONFIG_DIR, config_filename))

if __name__ == '__main__':
    
    preprocess_script_filename_list = config['SCRIPT']
        
    for preprocess_script_filename in preprocess_script_filename_list:
                
        # run script
        script_path = os.path.join(SCRIPT_DIR, preprocess_script_filename)
        command = f"python {script_path}"
        subprocess.call(command, shell=True)