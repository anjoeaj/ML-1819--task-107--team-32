# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:06:36 2018

@author: gargav
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedFold
from sklearn.metrics import accuracy_score

df = pd.read_csv("words_dataset.csv")

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['gender'],
                                                    test_size=0.25)

# setting up count vectorizer
vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=2,
                             max_features=1000)
vectorized_X = vectorizer.fit_transform(X_train)

print(vectorized_X.toarray())

vectorized_X_test = vectorizer.fit_transform(X_test)

params = {

    # depth of tree
    "max_depth": [4, 6],

    # min sum of weights
    # should be high enough to prevent over fitting
    # but not too high for over fitting
    "min_child_weight": [1, 5, 10],

    # the min loss value require to split
    "gamma": [0.5, 0.70, 1, 1.5, 2],

    # fraction of observations to be included in each tree
    # generally varies from 0.5-1
    "subsample": [0.75, 0.95],

    # fraction of column to be randomly sample in each tree
    "colsample_bytree": [0.6, 0.95]
}

"""
data_dmatrix = xgb.DMatrix(data=vectorized_X, label = y_train)

# difference between xgb.train and XGBCLassifier ????????  
booster = xgb.train(params, data_dmatrix)

# booster.save_model('0001.model')

dtest = xgb.DMatrix(vectorized_X_test.toarray())
"""

clf = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.95,
                        gamma=0.7, learning_rate=0.1, max_delta_step=0, max_depth=6,
                        min_child_weight=10, missing=None, n_estimators=100, nthread=-1,
                        objective='binary:logistic', reg_alpha=2e-05, reg_lambda=1,
                        scale_pos_weight=1, seed=0, silent=True, subsample=0.75)

gridsearch = GridSearchCV(clf, params)

clf.fit(vectorized_X, y_train)

ypred = clf.predict(vectorized_X_test)
print(ypred)
print(accuracy_score(y_train, ypred))



