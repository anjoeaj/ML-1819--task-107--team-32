# -*- coding: utf-8 -*-
"""
Logistic Regression
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

from sklearn import metrics
from sklearn.metrics import roc_curve, auc

#Import processed text
dataset = pd.read_csv("words_dataset.csv", ',')

#max word count
maxWordCount = 1000

#Import statistical features
dataset2 = pd.read_csv("stats_crossval_dataset.csv", ',')
features = dataset2[['fav_number', 'tweet_count', 'created', 'descLen', 'nameLen']]

# Separate Y column
gender = dataset['gender']
y_train, y_test = train_test_split(gender, test_size = 0.25, random_state = 0)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Bag of Words parameters
bow_vectorizer_text = CountVectorizer(max_df = 0.90, min_df = 2, max_features = maxWordCount, stop_words='english')

# Bag of Words object
bow_text = bow_vectorizer_text.fit_transform(dataset["text"])
for i in range(maxWordCount): 
    features.loc[:, i] = np.zeros(12894)

for i in range(maxWordCount): 
    for j in range(12894):
        features.loc[j][i+5] = bow_text[j, i]

#X_train = bow_text[:12894, :]
X_train, X_test = train_test_split(features, test_size = 0.25, random_state = 0)

# GridSearch
gridModel = LogisticRegression()
trim_parameter_list = [{'C': [0.01, 0.1, 1, 10, 100, 1000], "penalty":["l1","l2"]}]
gridsearch = model_selection.GridSearchCV(gridModel, trim_parameter_list)
gridsearch.fit(X_train, y_train)
print(gridsearch.best_params_)
print(gridsearch.score)

# Model
model = LogisticRegression(C = 10, penalty = "l1" )
model.fit(X_train, y_train) # training the model
print(model.score(X_train, y_train))
Y_pred = model.predict(X_test)
print(model.score(X_test, y_test))

######################### PLOT ROC CURVE ###############################

# Plot the graph for test data
probs = model.predict_proba(X_train)
preds = probs[:,1]
fpr1, tpr1, threshold = metrics.roc_curve(y_train, preds)
roc_auc1 = metrics.auc(fpr1, tpr1)

#Get the true positives and false positives 
probs = model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# draw the graph 
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')

#plot test data ROC curve
plt.plot(fpr, tpr, 'darkorange', label = 'AUC = %0.2f' % roc_auc)

#plot training data ROC curve
plt.plot(fpr1, tpr1, 'r', label = 'AUC = %0.2f' % roc_auc1)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--',color='navy')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

######################### ROC CURVE END ###############################





