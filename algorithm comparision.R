# ========================================
# Multiple Hypothesis Testing
# Part 1: K-fold Cross-Validation Paired t-Test
# Part 2: Analysis of Variance (ANOVA) Test
# Part 3: Wilcoxon Signed Rank test
# ========================================

# Load the required R packages
#library('C50')
library('kernlab')
#library('caret')
library('e1071')
#library('stats')
library('cvTools')

#Predictions were found by using the codes provided.
#The configuration of the CNN used is:
#2 layer convolutional neural network
#filetrs 32 in first and 20 in the second
#500 nodes in the hidden layer (the first fully connected layer)
#detailed configuration can be found in the configuration file stored in the configuration folder

#testing was done using other configuration such as increasing the number of layers, increasing the number of filters
#and also increasing the number of fully connected layers and the nodes in them. Improvement with a few changes increased the 
#accuracy by atleast 5 percent.

# *****************************************
# Part 1: Analysis of Variance (ANOVA) Test
# *****************************************
print('ANOVA Test Results')
#cifar datasets
#CNN
cnn_acc = c(100-70.07,	100-71.1,	100-69.8,	100-70.22,	100-69.27)
#Naive Bayes
nb_acc = c(100-29.64,	100-29.61,	100-29.5,	100-29.56,	100-29.71)
#Extra Tree
et_acc = c(100-35.06,	100-35.01,	100-35.34,	100-35.31,	100-36)
#Ada Boost
adaBoost_acc = c(100-32,	100-32.57,	100-33.05,	100-31.63,	100-32.3)
#Random Forest
rf_acc = c(100-47.24,	100-47.43,	100-46.75,	100-47.56,	100-47.88)

errors_column <- c(cnn_acc, nb_acc, et_acc, adaBoost_acc, rf_acc)
classifier_column <- c(rep('CNN',5),rep('NB',5),rep('ETree',5),rep('AdaBoost',5),rep('RF',5))
errors.df <- data.frame(errors_column, classifier_column)
colnames(errors.df) <- c("error", "classifier")
errors.df$error <- as.numeric(errors.df$error)
performance_aov <- aov(error ~ classifier, data = errors.df)
summary(performance_aov)
# *****************************************
# Part 3: Wilcoxon Signed Rank test
# *****************************************


#cifar 10 dataset
#Errors = 100 - Accuracies
#CNN
cifar_cnn_acc = 100-71.63
#KNN
cifar_knn_acc = 100-10.16
#Naive Bayes
cifar_nb_acc = 100-35.74
#Random Forest
cifar_rf_acc = 100-48.29
#Extra Tree Classifier
cifar_etree_acc = 100-29.76
#AdaBoost Classifier
cifar_adaboost_acc = 100-32.18


#stl 10 dataset
#Errors = 100 - Accuracies
#CNN
stl_cnn_acc = 100-51.07
#KNN
stl_knn_acc = 100-27.06
#Naive Bayes
stl_nb_acc = 100-33.18
#Random Forest
stl_rf_acc = 100-42.26
#Extra Tree Classifier
stl_etree_acc = 100-32.38
#AdaBoost Classifier
stl_adaboost_acc = 100-27.95


#caltech 101 dataset
#Errors = 100 - Accuracies
#CNN
caltech_cnn_acc = 100-14.86
#KNN
caltech_knn_acc = 100-34.28
#Naive Bayes
caltech_nb_acc = 100-34.83
#Random Forest
caltech_rf_acc = 100-41.52
#Extra Tree Classifier
caltech_adaboost_acc = 100-24.98
#AdaBoost Classifier
caltech_etree_acc = 100-14.01


#Wilcoxon test between CNN and KNN
print("Wilcoxon test between CNN and KNN")
errors_column <- c(cifar_knn_acc, stl_knn_acc, caltech_knn_acc, cifar_cnn_acc, stl_cnn_acc, caltech_cnn_acc)
classifier_column <- c(rep('KNN',3),rep('CNN',3))
errors.df <- data.frame(errors_column, classifier_column)
colnames(errors.df) <- c("error", "classifier")
wilcox.test(formula = error ~ classifier, data = errors.df, paired = TRUE)


#classifier1_column <- c(cifar_knn_acc, stl_knn_acc, caltech_knn_acc, 34, 23)
#classifier2_column <- c(cifar_cnn_acc, stl_cnn_acc, caltech_cnn_acc, 56.0, 65.0)
#wilcox.test(x=classifier1_column, y=classifier2_column, alternative='less', paired = TRUE)

#Wilcoxon between for CNN and NB
print("Wilcoxon between for CNN and NB")
errors_column <- c(cifar_cnn_acc, stl_cnn_acc, caltech_cnn_acc, cifar_nb_acc, stl_nb_acc, caltech_nb_acc)
classifier_column <- c(rep('CNN',3),rep('NB',3))
errors.df <- data.frame(errors_column, classifier_column)
colnames(errors.df) <- c("error", "classifier")
wilcox.test(formula = error ~ classifier, data = errors.df, paired = TRUE)

#Wilcoxon test between CNN and RF
print("Wilcoxon test between CNN and RF")
errors_column <- c(cifar_cnn_acc, stl_cnn_acc, caltech_cnn_acc, cifar_rf_acc, stl_rf_acc, caltech_rf_acc)
classifier_column <- c(rep('CNN',3),rep('RF',3))
errors.df <- data.frame(errors_column, classifier_column)
colnames(errors.df) <- c("error", "classifier")
wilcox.test(formula = error ~ classifier, data = errors.df, paired = TRUE)

#Wilcoxon test between CNN and AdaBoost
print("Wilcoxon test between CNN and AdaBoost")
errors_column <- c(cifar_cnn_acc, stl_cnn_acc, caltech_cnn_acc, cifar_adaboost_acc, stl_adaboost_acc, caltech_adaboost_acc)
classifier_column <- c(rep('CNN',3),rep('AdaBoost',3))
errors.df <- data.frame(errors_column, classifier_column)
colnames(errors.df) <- c("error", "classifier")
wilcox.test(formula = error ~ classifier, data = errors.df, paired = TRUE)

#Wilcoxon test between CNN and ETree
print("Wilcoxon test between CNN and ETree")
errors_column <- c(cifar_cnn_acc, stl_cnn_acc, caltech_cnn_acc, cifar_etree_acc, stl_etree_acc, caltech_etree_acc)
classifier_column <- c(rep('CNN',3),rep('ETree',3))
errors.df <- data.frame(errors_column, classifier_column)
colnames(errors.df) <- c("error", "classifier")
wilcox.test(formula = error ~ classifier, data = errors.df, paired = TRUE)
