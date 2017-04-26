import sys
import os
import numpy as np

def read_all_images_labels(path_):
    """
        :param path_to_data: the file containing the binary images from the STL-10 dataset
        :return: an array containing all the images
        """
    
    X_tr, y_tr, X_te, y_te = None, None, None, None
    
    path_to_data = path_ + '/train_X.bin'
    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)
        
        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.
        
        X_tr = np.reshape(everything, (-1, 3, 96, 96))
        
        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        X_tr = np.transpose(X_tr, (0, 2, 3, 1)).astype("float")

    path_to_labels = path_ + '/train_y.bin'
    with open(path_to_labels, 'rb') as f:
        y_tr = np.fromfile(f, dtype=np.uint8)
        y_tr -= 1

    path_to_data = path_ + '/test_X.bin'
    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)
        
        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.
        
        X_te = np.reshape(everything, (-1, 3, 96, 96))
        
        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        X_te = np.transpose(X_te, (0, 2, 3, 1)).astype("float")

    path_to_labels = path_ + '/test_y.bin'
    with open(path_to_labels, 'rb') as f:
        y_te = np.fromfile(f, dtype=np.uint8)
        y_te -= 1

    return X_tr, y_tr, X_te, y_te


def read_stl_images(dir_path, num_training=4000, num_validation=1000, num_testing=8000):
    # path to the binary train file with image data
    
    stl10_dir = os.path.join(dir_path, 'code_data/datasets/stl10_binary')
    #stl10_dir = os.path.join(dir_path, 'stl10_binary')
    
    X_train, y_train, X_test, y_test = read_all_images_labels(stl10_dir)
    print X_train.shape
    print y_train.shape
    print X_test.shape
    print y_test.shape
    
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