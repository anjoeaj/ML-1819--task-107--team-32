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
from matplotlib import pyplot as plt
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

#Put hastags in a separate column (including hastags from description as well)
dataset['hashtag'] = dataset['text'].str.findall(r"#(\w+)")+ dataset['description'].str.findall(r"#(\w+)")
        
# Remove hashtags from original column
dataset['text'] = dataset['text'].str.replace(r"(#)(\w+)\b", "")
       
# Removing non-alpha characters
dataset['text'] = dataset['text'].str.replace("[^a-zA-Z#]", " ")

# Remove words with less than 3 letters - likely no significance       
dataset['text'] = dataset['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

# Tokenize tweet
tokenized_tweet = dataset['text'].apply(lambda x: x.split())
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
dataset['text'] = tokenized_tweet

# Cleaning description column (same as above)
dataset['description'] = dataset['description'].str.lower()

dataset['description'] = dataset['description'].str.replace(r'https?://t\.co/\S+', '')

dataset['description'] = dataset['description'].str.replace(r"(@)(\w+)\b", "")

dataset['description'] = dataset['description'].str.replace(r"(#)(\w+)\b", "")

dataset['description'] = dataset['description'].fillna("")

dataset['description'] = dataset['description'].str.replace("[^a-zA-Z#]", " ")

dataset['description'] = dataset['description'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

# Tokenize description
tokenized_desc = dataset['description'].apply(lambda x: x.split())
for i in range(len(tokenized_desc)):
    tokenized_desc[i] = ' '.join(tokenized_desc[i])
dataset['description'] = tokenized_desc


all_words = ' '.join([text for text in dataset['description']])

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#Split data into training and test sets
train, test = train_test_split(dataset, test_size = 0.25, random_state = 0)

#Save as csv file
train.to_csv("words_training_dataset.csv", ",")
test.to_csv("words_testing_dataset.csv", ",")
dataset.to_csv("words_crossval_dataset.csv", ",")


# Calculate the score:
# LSTM for sequential text(0.5) + XGBOOST for the description(0.15) 
# + other data created previously (0.15)