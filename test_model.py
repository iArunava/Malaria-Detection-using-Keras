import numpy as np
import matplotlib.pyplot as plt
import os
from config import *
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import classification_report

classes = ['Parasitized', 'Uninfected']

def test_model(path, on_test_batch=False, image_path=None):
    rn50 = load_model(path)
    print ('[INFO]Model Loaded Successfully!')

    if on_test_batch:
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

    else:
        if image_path == None:
            raise Exception('Path to image is None')

        img = plt.imread(image_path)
        img = resize(img, (64, 64, 3))
        img = img[np.newaxis, :]
        predict = rn50.predict(img)

        print ('The Image passed is of class: {}'.format(classes[np.argmax(predict[0])]))


    print ('[INFO]Model Evaluated Successfully!!')
