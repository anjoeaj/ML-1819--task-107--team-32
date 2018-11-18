# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 23:46:29 2018

@author: gargav
"""

#from gensim.models import KeyedVectors

import pandas as pd
from sklearn.model_selection import train_test_split

#lower-case letter

# Importing the dataset
dataset = pd.read_csv('E:\\twitter.csv', encoding="ISO-8859-1")

dataset = dataset[['gender', 'description', 'text']]

# male = 0, female = 1
dataset['gender'] = dataset['gender'].map({'female': 1, 'male': 0})


# Cleaning text column
dataset['text'] = dataset['text'].str.lower()

dataset['text'] = dataset['text'].str.replace(r'https?://t\.co/\S+', 
       'SHORTENED_URL')

dataset['text'] = dataset['text'].str.replace(r"(@)(\w+)\b", "TAGGED")

dataset['text'] = dataset['text'].str.replace(r"(#)(\w+)\b", "HASHTAG")
    
# Removing gibberish as well as special characters form text
# ^[^<>]+$
dataset['text'] = dataset['text'].str.replace(r'[^A-Za-z0-9,.\'-_? ]+', '')

# As we are going to tain a LSTM network
# putting , as a seperate word will make more sense
dataset['text'] = dataset['text'].str.replace(',', ' ,')
dataset['text'] = dataset['text'].str.replace('.', ' .')
dataset['text'] = dataset['text'].str.replace('?', ' ?')



# Cleaning description column
dataset['description'] = dataset['description'].str.lower()

dataset['description'] = dataset['description'].str.replace(r'https?://t\.co/\S+', 
       'SHORTENED_URL')

dataset['description'] = dataset['description'].str.replace(r"(@)(\w+)\b", 
       "TAGGED")

dataset['description'] = dataset['description'].str.replace(r"(#)(\w+)\b", 
        "HASHTAG")
dataset['description'] = dataset['description'].fillna("Not Available")
# Removing gibberish as well as some special characters form text
dataset['description'] = dataset['description'].str.replace(
        r'[^A-Za-z0-9,.\'-_? ]+', '')


#Split data into training and test sets
train, test = train_test_split(dataset, test_size = 0.25, random_state = 0)

#Save as csv file
train.to_csv("ML-1819--task-107--team-32_cleanedUpDataTrainingSet.csv", ",")
test.to_csv("ML-1819--task-107--team-32_cleanedUpDataTestSet.csv", ",")


# Calculate the score:
# LSTM for sequential text(0.5) + XGBOOST for the description(0.15) 
# + other data created previously (0.15)