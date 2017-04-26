import os
import sys
import json
import numpy as np
from code_data.datasets.cifar10_dataset import get_CIFAR10_data
from code_data.datasets.stl10_dataset import read_stl_images
from code_data.datasets.caltech101_dataset import read_caltech_images
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB

def create_model(model):
    if model=='random_forest':
        print 'random forest'
        return RandomForestClassifier(n_estimators = 200, criterion = 'gini', max_depth = None, min_samples_split = 2, min_samples_leaf = 1, max_features = 'auto', bootstrap = True, oob_score = True, n_jobs = 1, random_state = None, verbose = True)
    elif model=='ada_boost':
        print 'ada boost'
        return AdaBoostClassifier(base_estimator=None, n_estimators=100, learning_rate=0.8, algorithm='SAMME.R', random_state=None)
    elif model=='extra_tree_classifier':
        print 'extra tree classifier'
        return ExtraTreesClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=True, warm_start=False, class_weight=None)
    elif model=='naive_bayes':
        print 'naive bayes'
        return GaussianNB()
    else:
        print 'incorrect model'
        sys.exit()

def load_dataset(config):
    DIR = cwd = os.getcwd()
    if config['dataset']=='cifar10':
        data = get_CIFAR10_data(DIR, num_training=config['num_training'], num_validation=0, num_testing=config['num_testing'])
        for k, v in data.iteritems():
            print '%s: ' % k, v.shape
    elif config['dataset']=='stl10':
        data = read_stl_images(DIR, num_training=config['num_training'], num_validation=0, num_testing=config['num_testing'])
        for k, v in data.iteritems():
            print '%s: ' % k, v.shape
    elif config['dataset']=='caltech101':
        data = read_caltech_images(DIR, num_training=config['num_training'], num_validation=0, num_testing=config['num_testing'], input_dim=[3,32,32])
        for k, v in data.iteritems():
            print '%s: ' % k, v.shape
    else:
        print 'dataset not present'
        sys.exit()
    Xtr = data['X_train'].reshape(data['X_train'].shape[0],np.prod(data['X_train'].shape[1:]))
    Xte = data['X_test'].reshape(data['X_test'].shape[0],np.prod(data['X_test'].shape[1:]))
    Ytr = data['y_train']
    Yte = data['y_test']

    return Xtr, Ytr, Xte, Yte


def main():
    with open('config/classifier_config.json') as json_data_file:
        config = json.load(json_data_file)
    
    
    Xtr, Ytr, Xte, Yte = load_dataset(config)

    if config['model']=='knn':
        k=10
        print 'value of k is: ', k
        
        print 'Predicting on testing set using the training set for distance'
        class_prediction = np.zeros(len(Yte))
        for idx,te_data in enumerate(Xte):
            class_count = np.zeros(len(np.unique(Yte)))
            k_nearest_class = Ytr[np.sum((Xtr-te_data)**2, axis=1).argsort()[:k]]
            k_nearest_class_count = np.bincount(k_nearest_class)
            class_prediction[idx] = np.argmax(k_nearest_class_count)
            print 'idx', idx

        class_prediction = np.array(class_prediction)

        print 'Predictions:'
        print class_prediction
        
        print 'Accuracy on testing test'
        print sum(class_prediction==np.array(Yte))/float(len(Yte))
    else:
        model = create_model(config['model'])

        print 'Fitting the model on traning set'
        model = model.fit(Xtr,Ytr)
        
        print 'Predicting on testing set'
        class_prediction = model.predict(Xte)
        
        print 'Predictions:'
        print class_prediction
        
        
        print 'Accuracy on testing test'
        print sum(class_prediction==np.array(Yte))/float(len(Yte))


if __name__=='__main__':
    main()

