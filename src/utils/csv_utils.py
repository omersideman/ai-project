import os
import pandas as pd
from file_utils import create_dirs_if_not_exist


def split_classes(csv_path, spectrograms_path, out_path, class_names, high_th, low_th):
    class1_path = os.path.join(out_path, class_names[0]) + '/'
    class2_path = os.path.join(out_path, class_names[1]) + '/'

    # create directories
    create_dirs_if_not_exist(class1_path)
    create_dirs_if_not_exist(class2_path)

    df = pd.read_csv(csv_path)
    # iterate through rows
    for index, row in df.iterrows():
        # get popularity
        popularity = row['number_of_videos']
        # get spectrogram path
        spectrogram_path = os.path.join(
            spectrograms_path, f"{row['id']}.png")
        # get class
        if popularity >= high_th:
            class_path = class1_path
        elif popularity <= low_th:
            class_path = class2_path
        else:  # skip if popularity is in between
            continue
        # copy spectrogram to class directory
        os.system(f"cp {spectrogram_path} {class_path}")

    # print number of files in each class
    print(
        f"Number of files in {class_names[0]}: {len(os.listdir(class1_path))}")
    print(
        f"Number of files in {class_names[1]}: {len(os.listdir(class2_path))}")


def how_many(csv_path, column_name, value, where='above'):
    df = pd.read_csv(csv_path)
    if where == 'above':
        return len(df[df[column_name] >= value])
    elif where == 'below':
        return len(df[df[column_name] <= value])
    else:
        raise ValueError(f"Invalid value for where: {where}")
