from torch.utils.data import Dataset
from module.util import load_json
import cv2
import os

class ImageDataset(Dataset):
    """
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
        """
        이미지 처리 연산 추가
        """
        image = image.reshape(-1, 784)

        target = self.label_dict[image_filename]

        return image, target

# class ImageDataLoader(DataLoader):
#     def __init__(self):
#        super(ImageDataLoader, self).__init__()
    
    