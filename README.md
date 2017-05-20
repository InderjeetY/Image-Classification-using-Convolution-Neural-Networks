# Image Classification using Convolution Neural Networks and Comparison with Other Algorithms
This is a Configurable Convolutoin Neural Network.

DATASETS
There are 3 datasets whose information and scripts name are provided below. To download the datasets, keep these scripts in the folder in which you wish to download the data (in our case it was the dataset folder itself) and go to the command prompt and execute the command “./script_name.sh”


The 3 data sets are:

CIFAR-10 dataset.
- 10 mutually exclusive classes with 6000
images per class.
- 60000 images of 32x32 pixels of 3 bands.
- 50000 images for training data (49000 for training and 1000 for validation).
- 10000 images for testing data.
It can be downloaded by running the script get_cifar_datasets.sh


STL-10 dataset.
- 10 classes of image categories.
- 500 training and 800 test images per class.
- Image size of 96x96 pixels.
It can be downloaded by running the script get_stl_datasets.sh


CatTech 101 dataset
- 101 classes of image categories.
- About 50 images per class.
- Image are roughly 300x200 pixels.
It can be downloaded by running the script get_caltech_datasets.sh



You will need to install the following packages:
1. Compile the Cython extension: Convolutional Neural Networks require a very efficient implementation. So other than the naive implemetation there is an of the functionality using Cython (by Stanford cs231n instructors); you will need to compile the Cython extension before you can run the code. From the code_data directory, run the following command:
    "python setup.py build_ext --inplace"
2. Numpy: It can be installed using the following command "pip install numpy"
3. cPickle: It can be installed using the following command "sudo pip install cpickle"



The following are the options that can be configured:
- 	Input Dimension – This parameter handles the dimensions of the image set will be having.
- 	Number of Convolution Layers – This parameter helps to generate CNN models of different number of CONV layers.
- 	Training Data set – This parameter is used to predefine the size of training dataset.
- 	Validation Data set – This parameter is used to predefine the size of validation data set.
- 	Weight Scale – This parameter helps to scale the randomly initialized Weights.
- 	Regularization – To set the value of regularization constant for Loss Function.
- 	Filter Size – To set size of filter weight matrix at CONV layer for feature extraction.
- 	BatchNorm – This parameter accepts the Boolean values for presence or absence of Normalization layer in CNN.
- 	Number of Epochs – This parameter defines the number of iterations required to train CNN which randomly initialized weights every time.
- 	Batch Size – Value of this parameter handles number of desired batches for training dataset.
- 	Update Rule – This is used to handle Weight’s updating function.
- 	Use Pool – This is an array of size of number of CONV layers in the model with only Boolean values to define that whether pooling is to be applied at each CONV layer. 
- 	Number of Filters - This is an array of size of number of CONV layers in the model with values to define the number of filters at each layer of feature extraction  
- 	Learning Rate – This parameter defines the rate of training the data set.
- 	Momentum – This parameter is used to handle the running variance and mean of CONV layer.
- 	lr_decay – This parameter the constant at which learning rate decays.
- 	Hidden Layer Dimensions – This parameter handles the number of hidden layers and their dimensions in Fully Connected layer to map each image linearly.
- 	Verbose – Set to true for printing the errors for each iteration of training.
- 	eps – It is the value used in batch normalization which is added to the variance
- 	dataset – It has 3 possible values: {"cifar10", "stl10", "caltech101"}
Sample configs have been provided in the configuration folder. The configuration files name should be "cnn_config.json"



You then go to the folder where the file convolution_net_main.py is placed and execute it by using the following command: "python convolution_net_main.py"




There are otehr algorithms that have been implemented in the file other_classifier.py and it also has a configuration files.

You will need to install the following pacakage:
1. sklearn: It can be installed using the following command "pip install -U scikit-learn"

You then go to the folder where the file convolution_net_main.py is placed and execute it by using the following command: "python other_classifier.py"

- 	model – It has the following possible outputs: {"random_forest", "ada_boost", "knn", "extra_tree_classifier", "naive_bayes"}
- 	num_training – The number of training data 
- 	num_testing – The number of testing data
- 	dataset – It has 3 possible values: {"cifar10", "stl10", "caltech101"}
Sample configs have been provided in the configuration folder. The configuration files name should be "classifier_config.json"


STATISTICAL TESTS
The errors for the algorithm were stored and then the ANOVA and Wilcoxon statistical tests were performed.

- For ANOVA test we used the Cifar 10 dataset on CNN, Naive Bayes, Random Forest, Ada Boost and Extra Tree Classifier algorithms.
- For Wilcoxon test we used 3 datasets (Cifar 10, Stl 10 and Caltech 101) on CNN, K-Nearest Neighbours (K=10), Naive Bayes, Random Forest, Ada Boost and Extra Tree Classifier algorithms.
To execute the code for statistical test run the following command in command line after going to the place where the file is kept: Rscript algorithm_comparision.R
