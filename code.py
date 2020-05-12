import pickle
import numpy as np
from os import listdir
from os.path import isfile, join
import os
from keras.utils import np_utils

# Function to unpickle the dataset
def unpickle_all_data(directory):
    
    # Initialize the variables
    train = dict()
    test = dict()
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    
    # Iterate through all files that we want, train and test
    # Train is separated into batches
    for filename in listdir(directory):
        if isfile(join(directory, filename)):
            
            # The TRAIN data
            if 'data_batch' in filename:
                print('Handing file: %s' % filename)
                
                # Opent the file
                with open(directory + '/' + filename, 'rb') as fo:
                    data = pickle.load(fo, encoding='bytes')

                if 'data' not in train:
                    train['data'] = data[b'data']
                    train['labels'] = np.array(data[b'labels'])
                else:
                    train['data'] = np.concatenate((train['data'], data[b'data']))
                    train['labels'] = np.concatenate((train['labels'], data[b'labels']))
           
            # The TEST data
            elif 'test_batch' in filename:
                print('Handing file: %s' % filename)
                
                # Open the file
                with open(directory + '/' + filename, 'rb') as fo:
                    data = pickle.load(fo, encoding='bytes')
                
                test['data'] = data[b'data']
                test['labels'] = data[b'labels']
    

    # Manipulate the data to the propper format
    for image in train['data']:
        # train_x (50K x 32 x 32 x 3)
        train_x.append(np.transpose(np.reshape(image,(3, 32,32)),  (1,2,0)))
    # train_y (50K x 1)
    train_y = [label for label in train['labels']]
    
    for image in test['data']:
        # test_x (10Kx 32 x 32 x 3 )
        test_x.append(np.transpose(np.reshape(image,(3, 32,32)), (1,2,0)))
    # test_y (10k x 1)
    test_y = [label for label in test['labels']]
    
    # Transform the data to np array format
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    
    return (train_x, train_y), (test_x, test_y)
# /content/drive/My Drive/CS410 Deep Learning Theory/Assignments/Assignment 3/Code3.ipynb

def pickle_dump(x_train, y_train, x_test, y_train):
    with open('data/validation/test-x', 'wb') as handle:
        pickle.dump(x_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/validation/test-y', 'wb') as handle:
        pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/training/train-x', 'wb') as handle:
        pickle.dump(x_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/training/train-y', 'wb') as handle:
        pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done reading and writing files.")

# Run the function with and include the folder where the data are
curr_dir = os.getcwd()
print("\nWorking in %s"%curr_dir)
(x_train, y_train), (x_test, y_test) = unpickle_all_data(curr_dir + '/data/cifar-10-batches-py/')

pickle_dump(x_train, y_train, x_test, y_test)