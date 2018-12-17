# -*- coding: utf-8 -*-
"""
Logistic Regression
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from patsy import dmatrices

#Import processed text
dataset = pd.read_csv("words_dataset.csv", ',')

#Import statistical features
dataset2 = pd.read_csv("stats_crossval_dataset.csv", ',')
features = dataset2[['fav_number', 'tweet_count', 'created', 'descLen']]
#dataset2 = dataset2[["fav_number","tweet_count","created","descLen"]]
gender, features = dmatrices('gender ~ fav_number + tweet_count + created + descLen', dataset2, return_type="dataframe")

# Separate Y column
gender = dataset['gender']
y_train, y_test = train_test_split(gender, test_size = 0.25, random_state = 0)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Bag of Words parameters
bow_vectorizer_text = CountVectorizer(max_df = 0.90, min_df = 2, max_features = 2500, stop_words='english')

# Bag of Words object
bow_text = bow_vectorizer_text.fit_transform(dataset["text"])
for i in range(2500): 
    features[i] = np.zeros(12894)

for i in range(2500): 
    for j in range(12894):
        features[i][j] = bow_text[j, i]

#X_train = bow_text[:12894, :]
X_train, X_test = train_test_split(features, test_size = 0.25, random_state = 0)

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

#Smaller parameter lists were used, here is an example
model=LogisticRegression()
trim_parameter_list = [{'C': [0.01, 0.1, 1, 10, 100, 1000], "penalty":["l1","l2"]}]
gridsearch = model_selection.GridSearchCV(model, trim_parameter_list)
gridsearch.fit(X_train, y_train)
print(gridsearch.best_params_)
print(gridsearch.score)

lreg = LogisticRegression(C = 0.1, penalty = "l2" )
lreg.fit(X_train, y_train) # training the model
print(lreg.score(X_train, y_train))
Y_pred = lreg.predict(X_test)
print(lreg.score(X_test, y_test))

# C - 0.1, L2 (70.9, 63.9)