import os
import pickle
# from keras.utils import np_utils

import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
from keras import utils
from keras.datasets import cifar10

plt.style.use('seaborn-whitegrid')

def load_normalized_data():
    input_path = os.getcwd() + '/data/'
    # Training and validation files
    files = ['training/train-y', 'training/train-x',
            'validation/test-y', 'validation/test-x']

    # Load training labels
    with open(input_path+files[0], 'rb') as lbpath:
        y_train = pickle.load(lbpath, encoding='bytes')
    # Load training samples
    with open(input_path+files[1], 'rb') as imgpath:
        x_train = pickle.load(imgpath, encoding='bytes')
    # Load validation labels
    with open(input_path+files[2], 'rb') as lbpath:
        y_test = pickle.load(lbpath, encoding='bytes')
    # Load validation samples
    with open(input_path+files[3], 'rb') as imgpath:
        x_test = pickle.load(imgpath, encoding='bytes')

    # Transofrm them to a float32 type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize the inputs
    x_train /= 255
    x_test  /= 255

    # One-hot Encoding for the labels only
    num_classes = 10
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test  = np_utils.to_categorical(y_test,  num_classes)

    return x_train, y_train, x_test, y_test


def CNN(input_nodes, output_nodes, x_train, y_train, x_test, y_test, experiment):
    # Create the model
    model = Sequential()
    # model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_nodes, activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_nodes, activation='relu'))
    # model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    if experiment >= 3:
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    if experiment >= 5:
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(512, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(512, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    


    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_nodes))
    model.add(Activation('softmax'))

    opt_rms = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', 
                  optimizer=opt_rms, 
                  metrics=['accuracy']
                  )
    model.summary()

    # we can compare the performance with or without data augmentation
    epochs = 20
    batch_size = 64
    data_augmentation = False

    # nImg = y_train.shape[0]
    # nRow = y_train.shape[1]
    # nCol = y_train.shape[2]
    # # nClr = y_train.shape[3]
    # y_train = y_train.reshape(nImg, nRow*nCol)

    # nImg = x_test.shape[0]
    # nRow = x_test.shape[1]
    # nCol = x_test.shape[2]
    # nClr = x_test.shape[3]
    # x_test = x_test.reshape(nImg, nRow*nCol, nClr)

    if experiment != 4:
        print('Not using data augmentation.')
        trained_model = model.fit(x_train, y_train, batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            shuffle=True
                            )
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by dataset std
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                zca_epsilon=1e-06,  # epsilon for ZCA whitening
                rotation_range=0,  # randomly rotate images in 0 to 180 degrees
                width_shift_range=0.1,  # randomly shift images horizontally
                height_shift_range=0.1,  # randomly shift images vertically
                shear_range=0.,  # set range for random shear
                zoom_range=0.,  # set range for random zoom
                channel_shift_range=0.,  # set range for random channel shifts
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                cval=0.,  # value used for fill_mode = "constant"
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False,  # randomly flip images
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0
                )

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        trained_model = model.fit_generator(
                            datagen.flow(x_train, y_train, batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            workers=4
                            )



    # if experiment == 4:
    #     # data augmentation
    #     datagen = ImageDataGenerator(
    #                     featurewise_center=False,  # set input mean to 0 over the dataset
    #                     samplewise_center=False,  # set each sample mean to 0
    #                     featurewise_std_normalization=False,  # divide inputs by std of the dataset
    #                     samplewise_std_normalization=False,  # divide each input by its std
    #                     zca_whitening=False,  # apply ZCA whitening
    #                     rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    #                     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    #                     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    #                     horizontal_flip=True,  # randomly flip images
    #                     vertical_flip=False  # randomly flip images
    #                     )


    # # Compute quantities required for feature-wise normalization
    # # (std, mean, and principal components if ZCA whitening is applied).
    # datagen.fit(x_train)


    # Compile the model


    

    # Fit the model on the batches generated by datagen.flow().
    # model.fit_generator(
    #                     datagen.flow(x_train, y_train, batch_size=batch_size),
    #                     epochs=epochs,
    #                     validation_data=(x_test, y_test)
    #                     )
    
    # evaluate the accuracy
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)

    activation_name = 'relu'

    print("\nAccuracy over test set: ", test_accuracy)
    print("Loss over the test set: ", test_loss)
    plotAccuracy(trained_model, str(experiment) )
    plotLoss(trained_model, str(experiment))


    # model_name='cifar-10-cnn-'+str(experiment)
    # model_path = os.getcwd() + "/model/"
    # model.save(model_path + model_name+'.hd5') # Keras model
    print("Saved Keras model")

def plotAccuracy(model, activation_name):
    """This plots the training and validation accuracy
    
    Arguments:
        model  -- The model with the accuracies that will be plotted
    """
    
    # get the accuracies from the model
    plt.plot(model.history['accuracy'],    'blue', label='Training')
    plt.plot(model.history['val_accuracy'], 'red', label='Validation')
    # prepare the plot
    plt.title('Accuracy for ' + activation_name)
    plt.ylabel('Accuracy (%)')
    plt.xlabel('epochs')
    plt.legend(loc='lower right')
    # plt.ylim(0,1.05)
    #show plot
    # plt.show()
    plt.savefig(activation_name + "accurcy" + ".png")
    plt.close()

def plotLoss(model, activation_name):
    plt.plot(model.history['loss'], 'magenta')
    plt.plot(model.history['val_loss'], 'green')
    plt.title('Loss for '  + activation_name)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    # plt.show()
    plt.savefig(activation_name +"loss"+".png")
    plt.close()


def load_normalized_data2():
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

    return x_train, y_train, x_test, y_test

def main():
    # load data
    x_train, y_train, x_test, y_test = load_normalized_data2()
    
    num_of_inputs = x_train.shape[1:]
    num_of_outputs = len(y_train[0])
    
    experiment = [2, 3, 4, 5]

    # build model
    for e in experiment:
        CNN(num_of_inputs, num_of_outputs, x_train, y_train, x_test, y_test, e)




if __name__ == "__main__":
    main()