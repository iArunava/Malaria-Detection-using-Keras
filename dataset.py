import os
import numpy as np
import shutil
from utils import *
from config import *

def create_datasets():
    images = list_images_from_path(DATASET_PATH)

    np.random.shuffle(images)

    train_split_till = int(len(images) * TRAIN_SPLIT)
    train_images = images[:train_split_till]
    test_images = images[train_split_till:]

    val_split_till = int(len(train_images) * VAL_SPLIT)
    val_images = train_images[:val_split_till]
    train_images = train_images[val_split_till:]

    print ('# Train Images: {}/{}'.format(len(train_images), len(images)),
           '\n# Validation Images: {}/{}'.format(len(val_images), len(images)),
           '\n# Test Images: {}/{}'.format(len(test_images), len(images)))

    datasets = [
        ('train', train_images, TRAIN_PATH),
        ('val', val_images, VAL_PATH),
        ('test', test_images, TEST_PATH)
    ]

    # Loop over datasets
    for (d_type, image_paths, base_path) in datasets:
        print ('[INFO]Building {} split'.format(d_type))

        if not os.path.exists(base_path):
            print ('[INFO] Creating {} directory'.format(base_path))
            os.makedirs(base_path)

        for inp_path in image_paths:
            filename = inp_path.split('/')[-1]
            label = inp_path.split('/')[-2]

            label_path = os.path.sep.join([base_path, label])

            if not os.path.exists(label_path):
                print ('[INFO] Creating {} directory'.format(label_path))
                os.makedirs(label_path)

            path_to_new_image = os.path.sep.join([label_path, filename])
            shutil.copy2(inp_path, path_to_new_image)

    print ('[INFO]Generated Train Validation and Test Splits Successfully!!')
