from shutil import copy
from util import save_json
from glob import glob
import os

def merge_data():
    data_dir = '/workspace/template_project/archive/mnist/'
    label_list = list(range(10))

    for label in label_list:
        train_label_dir = os.path.join(data_dir+'train/', str(label)+'/')
        test_label_dir = os.path.join(data_dir+'test/', str(label)+'/')
        
        merge_label_dir = os.path.join(data_dir+'merge/', str(label)+'/')

        if not(os.path.isdir(merge_label_dir)):
            os.mkdir(merge_label_dir)

        train_img_path_list = glob(os.path.join(train_label_dir, '**.png'), recursive=True)
        test_img_path_list = glob(os.path.join(test_label_dir, '**.png'), recursive=True)

        for img_path in train_img_path_list:
            img_file_name = img_path.split('/')[-1]    
            copy(img_path, os.path.join(merge_label_dir, str(label)+'_'+img_file_name))

        for img_path in test_img_path_list:
            img_file_name = img_path.split('/')[-1]    
            copy(img_path, os.path.join(merge_label_dir, str(label)+'_'+img_file_name))

        merge_img_path_list = glob(os.path.join(merge_label_dir, '**.png'), recursive=True)
        n_train, n_test, n_merge = len(train_img_path_list), len(test_img_path_list), len(merge_img_path_list)

        train_img_filename_list = set(os.listdir(train_label_dir))
        test_img_filename_list = set(os.listdir(test_label_dir))
        inter = train_img_filename_list.intersection(test_img_filename_list)
        n_inter = len(inter)
        
        check_sum = True if n_train + n_test - n_inter == n_merge else False

        msg = f"Label {label} Train: {n_train} Test: {n_test} Merge: {n_merge} Intersection: {n_inter} checksum: {check_sum}"

        print(msg)


def set_label():
    data_dir = '/workspace/template_project/archive/mnist/merge/'
    data_label_dir = '/workspace/template_project/archive/mnist/merge_labeled/image/'

    label_dict = dict()  # key: filename, value: label    
    label_list = range(10)

    for label in label_list:
        label_data_dir = os.path.join(data_dir, str(label))
        filename_list = os.listdir(label_data_dir)
        filepath_list = glob(label_data_dir + "/**.png", recursive=True) 
        # Add label to label dict
        for filename in filename_list:
            label_dict[filename] = label
            
        # Move file to img directory
        for filepath in filepath_list:
            filename = filepath.split('/')[-1]
            copy(filepath, data_label_dir+filename)
    
    save_json('/workspace/template_project/archive/mnist/merge_labeled/label.json', label_dict)

if __name__ == '__main__':
    set_label()