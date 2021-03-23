"""preprocesser

"""

import os, sys
sys.path.append('/workspace/template_project/image_classification')

# from module.dataloader import ImageDataset
from torch.utils.data import DataLoader
import numpy as np
import cv2

TRAIN_DATA_DIR = '/workspace/template_project/image_classification/data/sample10_splitted/train'

def scale(image):
    min_pixel, max_pixel = image.min(), image.max()
    image = 255 * (image-min_pixel) / (max_pixel - min_pixel)
    return image

def flatten(image):
    image = image.reshape(-1, 784)
    return image

if __name__ == '__main__':
    pass
