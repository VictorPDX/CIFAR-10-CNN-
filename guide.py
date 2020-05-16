#!/usr/bin/env python3

from __future__ import print_function
import os, sys, json, traceback, gzip
import pickle
import numpy as np

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.utils import multi_gpu_model
from keras.utils import np_utils

from keras.datasets import cifar10

# SageMaker paths
prefix      = '/opt/ml/'
input_path  = prefix + 'input/data/'
output_path = os.path.join(prefix, 'output')
model_path  = os.path.join(prefix, 'model')
param_path  = os.path.join(prefix, 'input/config/hyperparameters.json')
data_path   = os.path.join(prefix, 'input/config/inputdataconfig.json')

# Load MNIST data copied by SageMaker
def load_data(input_path):
    # # Adapted from https://github.com/keras-team/keras/blob/master/keras/datasets/fashion_mnist.py

    # # Training and validation files
    # files = ['training/train-y', 'training/train-x',
    #          'validation/test-y', 'validation/test-x']
    # # Load training labels
    # with open(input_path+files[0], 'rb') as lbpath:
    #     y_train = pickle.load(lbpath, encoding='bytes')
    # # Load training samples
    # with open(input_path+files[1], 'rb') as imgpath:
    #     x_train = pickle.load(imgpath, encoding='bytes')
    # # Load validation labels
    # with open(input_path+files[2], 'rb') as lbpath:
    #     y_test = pickle.load(lbpath, encoding='bytes')
    # # Load validation samples
    # with open(input_path+files[3], 'rb') as imgpath:
    #     x_test = pickle.load(imgpath, encoding='bytes')
    # print("Files loaded")

    # cifar10_data = keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    num_classes = 10
    # y_train = utils.to_categorical(y_train, num_classes)
    # y_test = utils.to_categorical(y_test, num_classes)
     # Transofrm them to a float32 type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize the inputs
    x_train /= 255
    x_test  /= 255

    # One-hot Encoding for the labels only
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test  = np_utils.to_categorical(y_test,  num_classes)


    return (x_train, y_train), (x_test, y_test)

# Main code
try:
    # Read hyper parameters passed by SageMaker
    # with open(param_path, 'r') as params:
    #     hyperParams = json.load(params)
    # print("Hyper parameters: " + str(hyperParams))
    
    # lr = float(hyperParams.get('lr', '0.001'))
    # batch_size = int(hyperParams.get('batch_size', '64'))
    # epochs = int(hyperParams.get('epochs', '100'))
    # gpu_count = int(hyperParams.get('gpu_count', '1'))
    epochs = 10
    batch_size = 64
    num_classes = 10
               
    # Read input data config passed by SageMaker
    # with open(data_path, 'r') as params:
    #     inputParams = json.load(params)
    # print("Input parameters: " + str(inputParams))

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = load_data(input_path)

    # One-hot Encoding
    num_classes = 10
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Create the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    print("There are %s" % num_classes )

    # if gpu_count > 1:
    #     model = multi_gpu_model(model, gpus=gpu_count)
    
    # Compile the model
    opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    # Transofrm them to a float32 type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize the input 
    x_train /= 255
    x_test /= 255
    
    #data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    model_name='cifar-10-cnn-'+str(epochs)
    model.save(model_path+'/'+model_name+'.hd5') # Keras model
    print("Saved Keras model")

    sys.exit(0)
except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        # with open(os.path.join(output_path, 'failure'), 'w') as s:
            # s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)
