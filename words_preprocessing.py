# -*- coding: utf-8 -*-

"""
Created on Sat Nov 17 23:46:29 2018
@author: gargav
"""

# from gensim.models import KeyedVectors


import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download()


lemmatizer = WordNetLemmatizer()


def nltk2wn_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    
    elif nltk_tag.startswith('V'):
        return wordnet.VERB

    elif nltk_tag.startswith('N'):
        return wordnet.NOUN

    elif nltk_tag.startswith('R'):
        return wordnet.ADV

    else:
        return None


def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))
    return " ".join(res_words)


# lower-case letter


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

#  - - - - - Cleaning text column - - - - -
dataset['text'] = dataset['text'].str.lower()

#Removing website links
dataset['text'] = dataset['text'].str.replace(r'https?://t\.co/\S+','')

# Removing tags ('@')
dataset['text'] = dataset['text'].str.replace(r"(@)(\w+)\b", "")

#Put hastags in a separate column (including hastags from description as well)
hashtags = dataset['text'].str.findall(r"#(\w+)")+ dataset['description'].str.findall(r"#(\w+)")
       
# Remove hashtags from original column
dataset['text'] = dataset['text'].str.replace(r"(#)(\w+)\b", "")
       
# Removing non-alpha characters
dataset['text'] = dataset['text'].str.replace("[^a-zA-Z#]", " ")

# Remove words with less than 3 letters - likely no significance       
dataset['text'] = dataset['text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

print(dataset['text'][1:4])
dataset['text'] = dataset['text'].apply(lemmatize_sentence)
print(dataset['text'][1:4])
# Tokenize tweet

# - - - - - Cleaning hashtag column - - - - -
hashtags = hashtags.fillna("")

for i in range(len(hashtags)):
    hashtags[i] = ' '.join(hashtags[i])
dataset['hashtags'] = hashtags


# - - - - - Cleaning description column (same as above) - - - - -
dataset['description'] = dataset['description'].str.lower()

dataset['description'] = dataset['description'].str.replace(r'https?://t\.co/\S+', '')

dataset['description'] = dataset['description'].str.replace(r"(@)(\w+)\b", "")

dataset['description'] = dataset['description'].str.replace(r"(#)(\w+)\b", "")

dataset['description'] = dataset['description'].fillna("")

dataset['description'] = dataset['description'].str.replace("[^a-zA-Z#]", " ")

dataset['description'] = dataset['description'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))


#dataset['description'] = dataset['description'].apply(lambda x: [lemmatize_sentence(i) for i in x])
print(dataset['description'][1:5])
dataset['description'] = dataset['description'].apply(lemmatize_sentence)
print(dataset['description'][1:5])

# print(dataset['description'][1:2])

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(dataset['text'])

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Split data into training and test sets

train, test = train_test_split(dataset, test_size=0.25, random_state=0)

# Save as csv file

train.to_csv("words_training_dataset.csv", ",")
test.to_csv("words_testing_dataset.csv", ",")
dataset.to_csv("words_crossval_dataset.csv", ",")
