"""Preprocessor
    * 210618
        - Add copy and paste augmentation
"""

import numpy as np
import cv2

def get_preprocessor(processor: str):
    
    if processor == 'normalize':
        return normalize_image

    elif processor == 'flatten':
        return np.ravel

    elif processor == 'normalize_histogram':
        return normalize_histogram

    elif processor == 'equalize_histogram':
        return equalize_histogram

    elif processor == 'adaptive_equalize_histogram':
        return adaptive_equalize_histogram
    
    elif processor == 'local_standardize':
        return local_standardize

    elif processor == 'local_standardize_positive':
        return local_standardize_positive

    elif processor == 'global_standardize':
        return global_standardize

    elif processor == 'global_standardize_positive':
        return global_stadardize_positive

    elif processor == 'imagenet_standardize':
        scaler = CustomStandardize(means=np.array([0.485, 0.456, 0.406]), stds=np.array([0.229, 0.224, 0.225])) # RGB

        return scaler.standardize

    elif processor == 'top_standardize':
        scaler = CustomStandardize(means=np.array([0.23556308, 0.23728161, 0.23364305]), stds=np.array([0.23061243, 0.22843244, 0.2301831]))

        return scaler.standardize

    else:
        return None

class CustomStandardize:
    
    def __init__(self, means: list, stds: list):
        # BGR
        self.means = means
        self.stds = stds
        self.max_pixel_value = 255

    def standardize(self, image):
        image = image / self.max_pixel_value
        standardized_image = (image - self.means) / self.stds
        return standardized_image
        
def local_standardize(image):
    """
    Standardize
    https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
    """
    means: np.array = image.mean(axis=(0, 1))
    stds: np.array = image.std(axis=(0, 1))
    standardized_image = (image - means) / stds

    return standardized_image


def local_standardize_positive(image):
    """
    Standardize with positive domain
    https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
    """
    standardized_image = local_standardize(image)
    standardized_image = np.clip(standardized_image, -1.0, 1.0)
    standardized_image = (standardized_image + 1.0) / 2.0
    return standardized_image


def global_standardize(image):
    """
    Standardize
    https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
    """
    mean, std = image.mean(), image.std()
    standardized_image = (image - mean) / std
    return standardized_image


def global_stadardize_positive(image):
    """
    Standardize with positive domain
    https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
    """
    standardized_image = global_standardize(image)
    standardized_image = np.clip(standardized_image, -1.0, 1.0)
    standardized_image = (standardized_image + 1.0) / 2.0
    return standardized_image


def normalize_image(image: np.ndarray, max_pixel_value:int = 255)->np.ndarray:
    """Normalize image by pixel
    """
    normalized_image = image / max_pixel_value

    return normalized_image


def normalize_histogram(image: np.ndarray)-> np.ndarray:
    """Normalize histogram
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab_image[:, :, 0] = cv2.normalize(lab_image[:, :, 0], None, 0, 255 , cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    histogram_normalized_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    return histogram_normalized_image


def equalize_histogram(image: np.ndarray)->np.ndarray:
    """Equalize histogram
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab_image[:, :, 0] = cv2.equalizeHist(lab_image[:, :, 0])  # l channel
    equalized_histogram_image = cv2.cvtColor(lab_image, cv2.COLOR_YUV2RGB)
    return equalized_histogram_image


def adaptive_equalize_histogram(image: np.ndarray)->np.ndarray:
    """Apply CLAHE equalize histogram
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab_image[:, :, 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab_image[:, :, 0])
    histogram_equalized_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    return histogram_equalized_image