# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 23:46:29 2018

@author: gargav
"""

"""ADD PERCENTAGE OF MALES FEMALES IN THE DATASET"""

#from gensim.models import KeyedVectors

import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
lemma = nltk.wordnet.WordNetLemmatizer()

#lower-case letter

# Importing the dataset
dataset = pd.read_csv('twitter.csv', encoding="ISO-8859-1")

dataset = dataset[['gender', 'description', 'text']]

# Gender data contains: male, female, unknown, brand
# Remove brand and unknown rows
valid = ["male", "female"]
dataset = dataset[dataset['gender'].isin(valid)]
dataset = dataset.reset_index(drop=True)

# Count how many are male and female in dataset
count = dataset['gender'].value_counts()
print(count)

# male = 0, female = 1
dataset['gender'] = dataset['gender'].map({'female': 1, 'male': 0})


# Cleaning text column
dataset['text'] = dataset['text'].str.lower()

#Removing website links
dataset['text'] = dataset['text'].str.replace(r'https?://t\.co/\S+','')

# Removing tags ('@')
dataset['text'] = dataset['text'].str.replace(r"(@)(\w+)\b", "")

#Put hastags in a separate column
dataset['hashtag'] = dataset['text'].str.findall(r"#(\w+)")+ dataset['description'].str.findall(r"#(\w+)")
        
# Remove hashtags from original column
dataset['text'] = dataset['text'].str.replace(r"(#)(\w+)\b", "")
       
# Removing gibberish as well as special characters form text
# ^[^<>]+$
dataset['text'] = dataset['text'].str.replace(r'[^A-Za-z0-9,.\'-? ]+', '')

# As we are going to tain a LSTM network
# putting , as a seperate word will make more sense
dataset['text'] = dataset['text'].str.replace(',', ' ,')
dataset['text'] = dataset['text'].str.replace('.', ' .')
dataset['text'] = dataset['text'].str.replace('?', ' ?')

dataset['text'] = dataset['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

dataset['text'] = dataset['text'].apply(lambda x: x.split())
dataset['text'].head()


dataset['text'] = dataset['text'].apply(lambda x: [lemma(i) for i in x]) # stemming
dataset['text'].head()
for i in range(len(dataset['text'])):
    dataset['text'][i] = ' '.join(dataset['text'][i])

# Cleaning description column
dataset['description'] = dataset['description'].str.lower()

dataset['description'] = dataset['description'].str.replace(r'https?://t\.co/\S+', '')

dataset['description'] = dataset['description'].str.replace(r"(@)(\w+)\b", "")

dataset['description'] = dataset['description'].str.replace(r"(#)(\w+)\b", "")

dataset['description'] = dataset['description'].fillna("Not Available")

# Removing gibberish as well as some special characters form text
dataset['description'] = dataset['description'].str.replace(r'[^A-Za-z0-9,.\'-? ]+', '')

dataset['description'] = dataset['description'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

dataset['description'] = dataset['description'].apply(lambda x: x.split())
dataset['description'].head()


#Split data into training and test sets
train, test = train_test_split(dataset, test_size = 0.25, random_state = 0)

#Save as csv file
train.to_csv("words_training_dataset.csv", ",")
test.to_csv("words_testing_dataset.csv", ",")
dataset.to_csv("words_crossval_dataset.csv", ",")


# Calculate the score:
# LSTM for sequential text(0.5) + XGBOOST for the description(0.15) 
# + other data created previously (0.15)