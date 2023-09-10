import os
from PIL import Image


def get_sample_image_size(directory_path):
    """
    Returns the size of the first image in the directory.
    :param directory_path: path to the directory containing images
    :return: size of the first image in the directory
    """
    image_path = os.path.join(directory_path, os.listdir(directory_path)[0])
    image = Image.open(image_path)
    num_channels = len(image.getbands())

    print(f"Number of channels in the image: {num_channels}")

    # flip the size because PIL returns (width, height) and cnn expects (height, width)
    return tuple(reversed(image.size))