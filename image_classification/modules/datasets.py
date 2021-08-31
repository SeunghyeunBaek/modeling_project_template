"""Dataset 정의
"""

from torch.utils.data import Dataset
from modules.utils import load_json
import numpy as np
import cv2
import os


class ImageDataset(Dataset):
    
    def __init__ (self, image_dir: str, label_path: str, preprocessors: list):
        self.image_dir = image_dir
        self.label = load_json(label_path)
        self.preprocessors = preprocessors
        self.filenames = sorted(list(self.label.keys()))
        

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        """
        Args:
            index(int):
        Returns:
            image(np.ndarray):
            target(int):
            filename(str):
        """
        filename = self.filenames[index]
        path_ = os.path.join(self.image_dir, filename)
        
        image = cv2.imread(path_, cv2.IMREAD_GRAYSCALE)
        target = self.label[filename]
        
        # Preprocessing
        for preprocess in self.preprocessors:
            image = preprocess(image)
        
        return image, target, filename

if __name__ == '__main__':
    pass