# -*- coding: utf-8 -*-
"""
SVM 
"""
from sklearn import svm
import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score


#Read twitter train data
dta = pd.read_csv("ML-1819--task-107--team-32_cleanedUpDataTrainingSet.csv", ',')


#Split columns to X and Y
y_train = dta.iloc[:,1]
X_train = dta.iloc[:, 2:-1]

#sReformat data
y_train, X_train = dmatrices('gender ~ created + fav_number + tweet_count + descLen + des_hashtag_count + nameLen + tweet_length + '
                'num_tagged + tweet_hashtags + C(has_mentioned_other_bio) + C(uses_default_link_color) + '
                'C(tweet_has_link)', dta, return_type="dataframe")

y_train = np.ravel(y_train)


# instantiate a logistic regression model, and fit with training data for X and y

modelPoly = svm.SVC(kernel = 'poly', degree = 3)
modelPoly = modelPoly.fit(X_train, y_train)

# check the accuracy on the training set
print(modelPoly.score(X_train, y_train))

#print(y_train.mean())

modelLinear = svm.SVC(kernel= 'rbf', degree= 5)
modelLinear = modelPoly.fit(X_train, y_train)

print(modelLinear.score(X_train, y_train))
# Test model on test data
#Y_pred = model.predict(X_test)
#print(model.score(X_test, Y_test))
