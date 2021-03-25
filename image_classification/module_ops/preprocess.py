"""
TODO:
    수정필요
"""

def scale(image):
    min_pixel, max_pixel = image.min(), image.max()
    image = 255 * (image-min_pixel) / (max_pixel - min_pixel)
    return image

def flatten(image):
    image = image.reshape(-1, 784)
    return image

def do_task(dict):
    """
    dict:
        'data'
        'config'
    """
    image = dict['data']
    image = scale(image)
    image = flatten(image)
    return dict