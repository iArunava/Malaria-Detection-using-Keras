import os
from os import listdir

def list_images_from_path(path):
    files = []

    if os.path.isfile(path):
        return files.append(path)

    if path[-1] != '/':
        path += '/'

    for file in listdir(path):
        if os.path.isfile(path + file):
            files.append(path + file)
        else:
            files += list_images_from_path(path + file)

    return files
