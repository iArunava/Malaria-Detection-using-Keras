import numpy as np
import os
from config import *
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import classification_report

def test_model(path):
    rn50 = load_model(path)
    print ('[INFO]Model Loaded Successfully!')

    val_aug = ImageDataGenerator(rescale=1.0/255)

    # Initialize the test generator
    test_gen = val_aug.flow_from_directory(
                    TEST_PATH,
                    class_mode="categorical",
                    target_size=(64, 64),
                    color_mode="rgb",
                    shuffle=False,
                    batch_size=BS)

    print ('[INFO]Evaluating the Model...')

    num_test_imgs = len(os.listdir(TEST_PATH + '/Uninfected')) + \
                    len(os.listdir(TEST_PATH + '/Parasitized'))

    test_gen.reset()
    pred_idxs = rn50.predict_generator(test_gen,
        steps=(num_test_imgs // BS) + 1)

    print (pred_idxs.shape, num_test_imgs)

    pred_idxs = np.argmax(pred_idxs, axis=1)

    print (classification_report(test_gen.classes, pred_idxs,
        target_names=test_gen.class_indices.keys()))

    print ('[INFO]Model Evaluated Successfully!!')
