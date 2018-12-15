import argparse
import os
from keras.preprocessing.image import ImageDataGenerator
from dataset import create_datasets
from ResNet50 import ResNet50
from test_model import *
from config import *

def lr_decay(epoch):
    epoch += 1
    power = 1.0
    alpha = LR * (1 - (epoch / float(EPOCHS))) ** power
    return alpha

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--train',
        type=bool,
        default=False,
        help='To train or test')

    parser.add_argument('-gp', '--graph-path',
        type=str,
        default='./models/trained_malaria_model.hdf5',
        help='The path where the model is stored.')

    parser.add_argument('-otb', '--on-test-batch',
        type=bool,
        default=False,
        help='Perform inference on the whole test set')

    parser.add_argument('-i', '--image',
        type=str,
        default=None,
        help='To the image file')

    FLAGS, unparsed = parser.parse_known_args()

    # Create all datasets
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(VAL_PATH) or \
       not os.path.exists(TEST_PATH):
        create_datasets()

    if not FLAGS.train:
        if FLAGS.image == None and FLAGS.on_test_batch == False:
            raise Exception('Path to image file is not provided!')
        elif FLAGS.on_test_batch:
            test_model(FLAGS.graph_path, FLAGS.on_test_batch)
        else:
            test_model(FLAGS.graph_path, FLAGS.on_test_batch, FLAGS.image)
        exit(0)

    # Initialize the ImageDataGenerator
    train_aug = ImageDataGenerator(
                    rescale=1.0 / 255,
                    rotation_range=20,
                    zoom_range=0.05,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    horizontal_flip=True,
                    fill_mode="nearest")

    val_aug = ImageDataGenerator(rescale=1.0/255)

    # Initialize the training generator
    train_gen = train_aug.flow_from_directory(
                    TRAIN_PATH,
                    class_mode="categorical",
                    target_size=(64, 64),
                    color_mode='rgb',
                    shuffle=True,
                    batch_size=BS)

    # Intialize the validation generator
    val_gen = val_aug.flow_from_directory(
                    VAL_PATH,
                    class_mode="categorical",
                    target_size=(64, 64),
                    color_mode="rgb",
                    shuffle=False,
                    batch_size=BS)

    # Initialize the test generator
    test_gen = val_aug.flow_from_directory(
                    TEST_PATH,
                    class_mode="categorical",
                    target_size=(64, 64),
                    color_mode="rgb",
                    shuffle=False,
                    batch_size=BS)

    # Training
    rn50 = ResNet50(input_shape=(64, 64, 3), classes=2)

    #opt = keras.optimizers.SGD(lr=LR, momentum=0.9)
    opt = keras.optimizers.Adam(lr=LR)

    rn50.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    fp = 'malaria-model-{epoch:02d}-{val_loss:.2f}.hdf5'

    callbacks = [#K.LearningRateScheduler(lr_decay, verbose=1),
                 K.ModelCheckpoint(filepath=fp,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=False,
                                   mode='auto')]

    H = rn50.fit_generator(
        train_gen,
        steps_per_epoch=len(train_images) // BS,
        validation_data=val_gen,
        validation_steps=len(val_images) // BS,
        epochs=EPOCHS,
        callbacks=callbacks)
