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
from xgboost import plot_tree
import graphviz
import matplotlib.pyplot as plt

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

# plot
clf_new = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                            colsample_bytree=0.6, gamma=1, learning_rate=0.01, max_delta_step=0,
                            max_depth=8, min_child_weight=1, missing=None, n_estimators=600,
                            n_jobs=1, nthread=1, objective='binary:logistic', random_state=0,
                            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                            silent=True, subsample=0.5)
clf_new.fit(X_train, y_train)
plot_tree(clf_new)

######################### PLOT ROC CURVE ###############################
from sklearn import metrics

# Plot the graph for test data
probs = clf_new.predict_proba(X_train)
preds = probs[:, 1]
fpr1, tpr1, threshold = metrics.roc_curve(y_train, preds)
roc_auc1 = metrics.auc(fpr1, tpr1)

# Get the true positives and false positives
probs = clf_new.predict_proba(X_test)
preds = probs[:, 1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# draw the graph
plt.title('Receiver Operating Characteristic')

# plot test data ROC curve
plt.plot(fpr, tpr, 'darkorange', label='Test AUC = %0.2f' % roc_auc)

# plot training data ROC curve
plt.plot(fpr1, tpr1, 'g', label='Train AUC = %0.2f' % roc_auc1)

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--', color='navy')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

######################### ROC CURVE END ###############################
