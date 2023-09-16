import os
import pandas as pd
from shutil import copyfile


def split_classes_threshold(csv_path, spectrograms_path, out_path, class_names, high_th, low_th):
    class1_path, class2_path = [os.path.join(
        out_path, name) for name in class_names]

    # Create directories if they don't exist
    for class_path in [class1_path, class2_path]:
        os.makedirs(class_path, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Copy spectrograms to class directories based on popularity
    for _, row in df.iterrows():
        popularity = row['number_of_videos']

        spectrogram_path = os.path.join(spectrograms_path, f"{row['id']}.png")
        class_path = class1_path if popularity >= high_th else (
            class2_path if popularity <= low_th else None)
        print(f"popularity: {popularity}, class_path: {class_path}")
        if class_path and os.path.exists(spectrogram_path):
            copyfile(spectrogram_path, os.path.join(
                class_path, os.path.basename(spectrogram_path)))

    # Print the number of files in each class
    for class_path, class_name in zip([class1_path, class2_path], class_names):
        file_count = len(os.listdir(class_path))
        print(f"Number of files in {class_name}: {file_count}")


def split_classes(csv_path, spectrograms_path, out_path, class_names, num_tracks):
    '''splits into directories of the first and last num_tracks tracks'''

    class1_path, class2_path = [os.path.join(
        out_path, name) for name in class_names]

    # Create directories if they don't exist
    for class_path in [class1_path, class2_path]:
        os.makedirs(class_path, exist_ok=True)

    df = pd.read_csv(csv_path)

    # get ids of first num_tracks and last num_tracks
    first_ids = df['id'][:num_tracks]
    last_ids = df['id'][-num_tracks:]

    first_png_paths = [os.path.join(
        spectrograms_path, f"{id}.png") for id in first_ids]
    last_png_paths = [os.path.join(
        spectrograms_path, f"{id}.png") for id in last_ids]

    for png_path in first_png_paths:
        if os.path.exists(png_path):
            copyfile(png_path, os.path.join(
                class1_path, os.path.basename(png_path)))

    for png_path in last_png_paths:
        if os.path.exists(png_path):
            copyfile(png_path, os.path.join(
                class2_path, os.path.basename(png_path)))

    # Print the number of files in each class
    for class_path, class_name in zip([class1_path, class2_path], class_names):
        file_count = len(os.listdir(class_path))
        print(f"Number of files in {class_name}: {file_count}")


def how_many(csv_path, column_name, value, where='above'):
    df = pd.read_csv(csv_path)
    if where == 'above':
        return len(df[df[column_name] >= value])
    elif where == 'below':
        return len(df[df[column_name] <= value])
    else:
        raise ValueError(f"Invalid value for where: {where}")
