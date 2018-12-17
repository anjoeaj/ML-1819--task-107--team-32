# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 19:47:43 2018

@author: gargav
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("words_dataset.csv")

# setting up count vectorizer
vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=2,
                             max_features=3000)
x = vectorizer.fit_transform(df['text'])

encoder = LabelEncoder()
y = encoder.fit_transform(df['gender'])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

params = {

    # depth of tree
    "max_depth": [4, 6, 8],

    # min sum of weights
    # should be high enough to prevent over fitting
    # but not too high for over fitting
    "min_child_weight": [1, 5, 10],

    # the min loss value require to split
    "gamma": [0.5, 0.70, 1, 1.5, 2],

    # fraction of observations to be included in each tree
    # generally varies from 0.5-1
    "subsample": [0.5, 0.75, 0.95],

    # fraction of column to be randomly sample in each tree
    "colsample_bytree": [0.45, 0.6, 0.95]
}

"""
data_dmatrix = xgb.DMatrix(data=vectorized_X, label = y_train)

# difference between xgb.train and XGBCLassifier ????????  
booster = xgb.train(params, data_dmatrix)

# booster.save_model('0001.model')

dtest = xgb.DMatrix(vectorized_X_test.toarray())
"""

clf = xgb.XGBClassifier(learning_rate=0.01, n_estimators=600,
                        objective='binary:logistic', silent=True, nthread=1)

folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

random_search = RandomizedSearchCV(clf, params, n_iter=param_comb,
                                   scoring='roc_auc', n_jobs=4,
                                   cv=skf.split(X_train, y_train),
                                   verbose=3, random_state=1001)

random_search.fit(X_train, y_train)

print(random_search.best_estimator_)
print(random_search.best_score_)

ypred = random_search.predict(X_test)

print(accuracy_score(y_test, ypred))


