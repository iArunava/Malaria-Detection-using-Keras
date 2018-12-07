import keras
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D
from keras.layers import BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, GlobalMaxPooling2D, AveragePooling2D
from keras.initializers import glorot_uniform
from keras.models import Model, load_model
import numpy as np

# Implementation of the identity block
def identity_block(X, f, filters, stage, block):

    # defining_name_basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve filters
    f1, f2, f3 = filters

    # Copy the input value for the skip branch
    X_copy = X

    # First component of main path
    X = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2a', kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=f2, kernel_size=(f, f), strides=(1, 1), padding='same',
               name=conv_name_base + '2b', kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component
    X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                name=conv_name_base + '2c', kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final Step
    X = Add()([X, X_copy])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2):
    # Defining base names
    conv_base_name = 'res' + str(stage) + block + '_branch'
    bn_base_name = 'bn' + str(stage) + block + '_branch'

    # Retrieve filters
    f1, f2, f3 = filters

    # Save copy for skip branch ops
    X_copy = X

    # First component - Main path
    X = Conv2D(filters=f1, kernel_size=(1, 1), strides=(s, s), padding='valid',
               name=conv_base_name + '2a', kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3, name=bn_base_name + '2a')(X)
    X = Activation('relu')(X)

    # Second component - Main path
    X = Conv2D(filters=f2, kernel_size=(f, f), strides=(1, 1), padding='same',
               name=conv_base_name + '2b', kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3, name=bn_base_name + '2b')(X)
    X = Activation('relu')(X)

    # Third Component - Main path
    X = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                name=conv_base_name + '2c', kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3, name=bn_base_name + '2c')(X)

    # First Component - Skip Path
    X_copy = Conv2D(filters=f3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                     name=conv_base_name + '1', kernel_initializer=glorot_uniform())(X_copy)
    X_copy = BatchNormalization(axis=3, name=bn_base_name + '1')(X_copy)

    # Add the shortcut
    X = Add()([X_copy, X])
    X = Activation('relu')(X)

    return X

def ResNet50(input_shape=(64, 64, 3), classes=6):
    # Define the Input Tensor
    X_inp = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_inp)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=1)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, filters=[128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=1)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, filters=[256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2, stage=5, block='a')
    X = identity_block(X, 3, filters=[512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, filters=[512, 512, 2048], stage=5, block='c')

    # AvgPool
    X = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

    # Output Layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform())(X)

    model = Model(inputs=X_inp, outputs=X, name='ResNet50')

    return model
