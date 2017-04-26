import sys
import os
import glob
import math
import numpy as np
from scipy import misc
from random import shuffle

def read_all_images_labels(path_, input_dim):
    """
        :param path_to_data: the file containing the binary images from the STL-10 dataset
        :return: an array containing all the images
        """
    
    X_tr, y_tr, X_te, y_te = [], [], [], []
    data_tr = []
    
    folders = glob.glob(os.path.join(path_, '*'))#os.listdir(path_)
    image_class = 0
    min_dim = 1000
    for folder in folders:
        path_to_image = os.path.join(path_,folder)
        images = glob.glob(os.path.join(path_to_image, '*'))#os.listdir(path_to_image)
        total_images = len(images)
        testing_count = int(math.ceil((len(images)*15)/100.0))
        training_count = total_images - testing_count
        image_count = 0
        for image in images:
            img_vec = misc.imread(os.path.join(path_to_image,image), mode='RGB')
            if min_dim>img_vec.shape[0]:
                min_dim=img_vec.shape[0]
            if min_dim>img_vec.shape[1]:
                min_dim=img_vec.shape[1]
            img_vec = misc.imresize(img_vec, tuple(input_dim[1:]+input_dim[0:1]))
            if image_count<testing_count:
                X_te.append(img_vec[np.newaxis,...].astype('float'))
                y_te.append(image_class)
                image_count += 1
            else:
                data_tr.append((img_vec[np.newaxis,...].astype('float'),image_class))
                image_count += 1
        image_class += 1

    print min_dim
    shuffle(data_tr)
    X_tr = [x for (x,y) in data_tr]
    y_tr = [y for (x,y) in data_tr]

    return np.concatenate(X_tr), np.array(y_tr), np.concatenate(X_te), np.array(y_te)


def read_caltech_images(dir_path='/Users/inderjeetyadav/Graduate/sem2/ALDA/Project/updating', num_training=6724, num_validation=1000, num_testing=1421, input_dim = [3,32,32]):
    # path to the binary train file with image data
    
    caltech101_dir = os.path.join(dir_path, 'code_data/datasets/101_ObjectCategories')
    #stl10_dir = os.path.join(dir_path, '101_ObjectCategories')
    
    X_train, y_train, X_test, y_test = read_all_images_labels(caltech101_dir, input_dim)

    X_val = X_train[range(num_training, num_training + num_validation)]
    y_val = y_train[range(num_training, num_training + num_validation)]
    X_train = X_train[range(num_training)]
    y_train = y_train[range(num_training)]
    X_test = X_test[range(num_testing)]
    y_test = y_test[range(num_testing)]
    
    print X_val.shape
    print y_val.shape
    print X_train.shape
    print y_train.shape
    print X_test.shape
    print y_test.shape
    
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