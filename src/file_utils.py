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
