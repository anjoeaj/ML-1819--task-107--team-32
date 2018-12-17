# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 18:51:43 2018

@author: cjmcm
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

#max word count
maxWordCount = 100

# Import words dataset
text_dataset = pd.read_csv("words_dataset.csv", ',')

#Import statistical features
dataset = pd.read_csv("stats_dataset.csv", ',')
features = dataset[['gender','fav_number', 'tweet_count', 'created', 'descLen', 'nameLen']]

# Bag of Words parameters
bow_vectorizer_text = CountVectorizer(max_df = 0.90, min_df = 2, max_features = maxWordCount, stop_words='english')

# Bag of Words object
bow_text = bow_vectorizer_text.fit_transform(text_dataset["text"])

# Add values to matrix (because of awkward formatting issue)
for i in range(maxWordCount): 
    features.loc[:, i] = np.zeros(12894)

for i in range(maxWordCount): 
    for j in range(12894):
        features.loc[j][i] = bow_text[j, i]
        
# Save features dataset        
features.to_csv("features_test.csv", ",")