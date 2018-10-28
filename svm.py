# -*- coding: utf-8 -*-
"""
SVM 
"""

import numpy as np
import pandas as pd
from patsy import dmatrices
#Read twitter train data
dtraining = pd.read_csv("ML-1819--task-107--team-32_cleanedUpDataTrainingSet.csv", ',')


#Split columns to X and Y
y_train = dtraining.iloc[:,1]
X_train = dtraining.iloc[:, 2:-1]

#Reformat data
y_train, X_train = dmatrices('gender ~ created + fav_number + tweet_count + descLen + des_hashtag_count + nameLen + tweet_length + '
                'num_tagged + tweet_hashtags + C(has_mentioned_other_bio) + C(uses_default_link_color) + '
                'C(tweet_has_link)', dtraining, return_type="dataframe")

y_train = np.ravel(y_train)


from sklearn import svm
from sklearn import model_selection
# instantiate an SVM model, and fit with training data for X and y
# svm.SVC (C=1.0, kernel= 'rbf’, degree=3, gamma=’auto’)
# gamma = 'auto' = 1/n features
# Can use sklearn GridSearchCV to optimise
# This list wasn't actually checked with GridSearch, instead I checked each kernel individually to segment the process 
"""parameter_list = [{'kernel':['linear'], 'C': [ 0.01, 0.1, 1, 10, 100, 1000]},
                   {'kernel':['rbf'], 'gamma': ['auto', 0.001, 0.01, 0.1, 0, 1, 10, 100, 1000], 'C': [0.01, 0.1, 1, 10, 100, 1000]},
                   {'kernel':['sigmoid'], 'gamma': ['auto', 0.001, 0.01, 0.1, 0, 1, 10, 100, 1000], 'C': [ 0.01, 0.1, 1, 10, 100, 1000]},
                   {'kernel':['poly'], 'gamma': ['auto', 0.001, 0.01, 0.1, 0, 1, 10, 100, 1000], 'C': [ 0.01, 0.1, 1, 10, 100, 1000], 'degree': [2, 3, 4]}]
"""

#Smaller parameter lists were used, here is an example
"""modelSVM = svm.SVC()
trim_parameter_list = [{'kernel':['linear'], 'C': [ 0.01, 0.1, 1, 10, 100, 1000]}]
gridsearch = model_selection.GridSearchCV(modelSVM, trim_parameter_list)
gridsearch.fit(X_train, y_train)
print(gridsearch.best_params_)
print(gridsearch.score)"""
# Best parameters were noted and then applied manually

# Specific tests
modelSVM = svm.SVC(C = 10,  kernel = "rbf", gamma = 1000)
modelSVM.fit(X_train, y_train)
print(modelSVM.score(X_train, y_train))


# Test model on test data
#Read twitter test data
dtest = pd.read_csv("ML-1819--task-107--team-32_cleanedUpDataTestSet.csv", ',')

#Split columns to X and Y
y_test = dtest.iloc[:,1]
X_test = dtest.iloc[:, 2:-1]

#Reformat data
y_test, X_test = dmatrices('gender ~ created + fav_number + tweet_count + descLen + des_hashtag_count + nameLen + tweet_length + '
                'num_tagged + tweet_hashtags + C(has_mentioned_other_bio) + C(uses_default_link_color) + '
                'C(tweet_has_link)', dtest, return_type="dataframe")

y_test = np.ravel(y_test)

Y_pred = modelSVM.predict(X_test)
print(modelSVM.score(X_test, y_test))








