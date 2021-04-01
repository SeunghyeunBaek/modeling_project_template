"""Dataset 정의

TODO:
    image_filename 추가

"""

from torch.utils.data import Dataset
from modules.utils import load_json
#from module_processor.preprocessor import scale, flatten
import cv2
import os


class ImageDataset(Dataset):
    """데이터셋 정의
    """
    
    def __init__ (self, image_dir: str, label_path: str):
        self.image_dir = image_dir
        self.label_dict = load_json(label_path)
        self.image_filename_list = sorted(list(self.label_dict.keys()))

    def __len__(self):
        return len(self.label_dict.keys())

    def __getitem__(self, index: int):
        image_filename = self.image_filename_list[index]
        image_path = os.path.join(self.image_dir, image_filename)
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Preprocessing
        # image = scale(image)
        # image = flatten(image)
        image = image.reshape(-1, 784)

        target = self.label_dict[image_filename]

        #return image, target
        # item_dict = {
        #     'image': image,
        #     'target': target,
        #     'image_filename': image_filename
        # }
        
        return image, target, image_filename

if __name__ == '__main__':
    pass