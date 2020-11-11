import numpy as np
from PIL import Image


def load_image(in_filename):
    """
    Load image
    :param in_filename: str
    Image to load
    :return: np.array
    Loaded image
    """
    img = Image.open(in_filename)
    img.load()
    data = np.asarray(img, dtype="uint8")
    return data

