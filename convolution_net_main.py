import os
import numpy as np
from code_data.classifiers.cnn import *
from code_data.datasets.cifar10_dataset import get_CIFAR10_data
from code_data.datasets.stl10_dataset import read_stl_images
from code_data.datasets.caltech101_dataset import read_caltech_images
from code_data.solver import Solver
import json
import sys
import math


########################################################################################################
def get_dataset(config):
    DIR = cwd = os.getcwd()
    if config['dataset']=='cifar10':
        data = get_CIFAR10_data(DIR, num_training=config['num_training'], num_validation=config['num_validation'], num_testing=config['num_testing'])
        for k, v in data.iteritems():
            print '%s: ' % k, v.shape
    elif config['dataset']=='stl10':
        data = read_stl_images(DIR, num_training=config['num_training'], num_validation=config['num_validation'], num_testing=config['num_testing'])
        for k, v in data.iteritems():
            print '%s: ' % k, v.shape
    elif config['dataset']=='caltech101':
        data = read_caltech_images(DIR, num_training=config['num_training'], num_validation=config['num_validation'], num_testing=config['num_testing'], input_dim=config['input_dim'])
        for k, v in data.iteritems():
            print '%s: ' % k, v.shape
    else:
        print 'dataset not present'
        sys.exit()
    return data



########################################################################################################
#to test the accuracy of a model
def check_accuracy(model, X, y, num_samples=None, batch_size=100):
    """
        Check accuracy of the model on the provided data.
        
        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
        on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using too
        much memory.
        
        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
        classified by the model.
        """
    
    # Maybe subsample the data
    N = X.shape[0]
    if num_samples is not None and N > num_samples:
        mask = np.random.choice(N, num_samples)
        N = num_samples
        X = X[mask]
        y = y[mask]
    
    # Compute predictions in batches
    num_batches = int(math.ceil(float(N) / batch_size))
    y_pred = []
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        scores = model.loss(X[start:end])
        y_pred.append(np.argmax(scores, axis=1))
    y_pred = np.hstack(y_pred)
    acc = np.mean(y_pred == y)

    return acc


def main():
    with open('config/cnn_config.json') as json_data_file:
        config = json.load(json_data_file)

    data = get_dataset(config)
    if len(config['num_filters'])!=config['num_of_conv_layers']:
        print 'incorrect number of filters or convolution layers'
        sys.exit()

    num_classes = len(np.unique(data['y_train']))+len(np.unique(data['y_val']))+len(np.unique(data['y_test']))

    model_2 = ConvNet(input_dim=config['input_dim'], weight_scale=config['weight_scale'], reg=config['reg'], filter_size = config['filter_size'],use_batchnorm=config['use_batchnorm'], num_of_conv_layers=config['num_of_conv_layers'], num_filters=config['num_filters'], momentum=config['momentum'], eps=config['eps'], num_classes=num_classes, use_pool=config['use_pool'], pool_size_stride=config['pool_size_stride'], hidden_layer_dimensions = config['hidden_layer_dimensions'])

    solver_2 = Solver(model_2, data, num_epochs=config['num_epochs'], batch_size=config['batch_size'], update_rule=config['update_rule'], optim_config=config['optim_config'], verbose=config['verbose'], print_every=config['print_every'], lr_decay=config['lr_decay'])
    solver_2.train()


    print 'accuracy'
    print check_accuracy(model_2, data['X_test'], data['y_test'])

if __name__=='__main__':
    main()