import glob
import pandas as pd
import os
import json
import os


def pretty_print(obj):
    print(json.dumps(obj, indent=4))


def create_dirs_if_not_exist(path):
    '''creates directories for either a dir path or a file path'''
    dirs, filename = os.path.split(path)
    os.makedirs(dirs, exist_ok=True)
    return os.path.join(dirs, filename)


def load_json(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r') as f:
        return json.load(f)


def merge_csv(input_dir, output_file):
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    df = pd.concat((pd.read_csv(f) for f in csv_files))
    df.to_csv(output_file, index=False)


def train_test_split(input_dir, output_dir, test_ratio):
    """splits a directory, containing two subdirectories of images
    representing two classes, into train and test sets"""
    os.makedirs(output_dir, exist_ok=True)

    training_path = os.path.join(output_dir, 'training')
    testing_path = os.path.join(output_dir, 'testing')

    # create subdirectories training and testing
    os.makedirs(training_path, exist_ok=True)
    os.makedirs(testing_path, exist_ok=True)

    # calculate how many testing images to take from each class
    # ignore non directories
    classes = [f for f in os.listdir(input_dir) if os.path.isdir(
        os.path.join(input_dir, f))]
    num_images_per_class = [num_images(os.path.join(input_dir, class_name))
                            for class_name in classes]
    print(f'Number of images per class: {num_images_per_class}')
    num_testing_images = [
        int(num * test_ratio) for num in num_images_per_class]
    print(
        f'Number of testing images per class: {num_testing_images}')

    for class_name, num_testing_images in zip(classes, num_testing_images):
        # create subdirectories for each class in training and testing
        class_training_dir = os.path.join(training_path, class_name)
        class_testing_dir = os.path.join(testing_path, class_name)
        os.makedirs(class_training_dir, exist_ok=True)
        os.makedirs(class_testing_dir, exist_ok=True)

        # move images from input_dir to training and testing
        class_dir = os.path.join(input_dir, class_name)
        class_files = os.listdir(class_dir)
        test_files = class_files[:num_testing_images]
        for f in test_files:
            os.rename(os.path.join(class_dir, f),
                      os.path.join(class_testing_dir, f))
        train_files = class_files[num_testing_images:]
        for f in train_files:
            os.rename(os.path.join(class_dir, f),
                      os.path.join(class_training_dir, f))

        return training_path, testing_path


def num_images(directory_path):
    png_files = [f for f in os.listdir(directory_path) if f.endswith('.png')]
    return len(png_files)
