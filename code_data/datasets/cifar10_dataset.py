import cPickle
import numpy as np
import os
import sys

def load_CIFAR10(dir):
    #getting training data
    x_ = []
    y_ = []
    for b in range(1, 6):
        file = os.path.join(dir, 'data_batch_%d' % (b, ))
        f = open(file, 'rb')
        datadict = cPickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        x_.append(X)
        y_.append(Y)
    X_tr = np.concatenate(x_)
    Y_tr = np.concatenate(y_)

    #getting testing data
    file = os.path.join(os.path.join(dir, 'test_batch'))
    f = open(file, 'rb')
    datadict = cPickle.load(f)
    X_te = datadict['data']
    Y_te = datadict['labels']
    X_te = X_te.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    Y_te = np.array(Y_te)

    return X_tr, Y_tr, X_te, Y_te


def get_CIFAR10_data(dir_path, num_training=49000, num_validation=1000, num_testing=10000):
    dir = os.path.join(dir_path, 'code_data/datasets/cifar-10-batches-py')
    X_train, y_train, X_test, y_test = load_CIFAR10(dir)
    
    X_val = X_train[range(num_training, num_training + num_validation)]
    y_val = y_train[range(num_training, num_training + num_validation)]
    X_train = X_train[range(num_training)]
    y_train = y_train[range(num_training)]
    X_test = X_test[range(num_testing)]
    y_test = y_test[range(num_testing)]

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }