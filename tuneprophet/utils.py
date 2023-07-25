import json
import os


def pretty_print(obj):
    print(json.dumps(obj, indent=4))


def create_dirs_if_not_exist(path):
    '''creates directories for either a dir path or a file path'''
    dirs, filename = os.path.split(path)
    os.makedirs(dirs, exist_ok=True)
    return os.path.join(dirs, filename)
