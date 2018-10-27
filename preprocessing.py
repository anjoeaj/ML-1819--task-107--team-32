# -*- coding: utf-8 -*-
"""
ML Twitter Data Preprocessing
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from datetime import date, datetime

# Determine how many years old each account is
def calculate_age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

# Function created to remove data not taken from twitter
def CleanUp(dataset):
    dataset = dataset.drop(columns=["_unit_id", "_golden", "_unit_state", "_trusted_judgments", "_last_judgment_at"])
    dataset = dataset.drop(columns=["gender:confidence", "profile_yn", "profile_yn:confidence", "gender_gold"])
    dataset = dataset.drop(columns=["profile_yn_gold", "profileimage"])
    return dataset

# Functino to normalize values in data columns
def normalizeCol(colName):
    colVals = X[colName].values.astype(float)
    norm = colVals/np.linalg.norm(colVals)
    return norm


# Importing the dataset
dataset = pd.read_csv('twitter.csv', encoding="ISO-8859-1")

# Cleaning the dataset (creators of dataset added features for their specific model)
dataset = CleanUp(dataset)

# y will be the gender of the account
y = dataset.iloc[:, 0].values

# x will be our modifiers
X = dataset.drop(columns="gender")

# Removing 'tweet coords' because majority are nan values
X = X.drop(columns="tweet_coord")

# Removing 'tweet ID' because values are random
X = X.drop(columns="tweet_id")

# Removing 'tweet time' because all tweets were captured within a two minute span
X = X.drop(columns="tweet_created")

# Removing tweet location because it is not relevant or quanitifiable
X = X.drop(columns="tweet_location")

# Removing retweet_count because of insufficient data / data seems implausible
X = X.drop(columns="retweet_count")

# Removing user timezone because majority of the data is NaN
X = X.drop(columns="user_timezone")

# Removing sidebar color because default is unknown and feature is deprecated
X = X.drop(columns="sidebar_color")


# In description columns (twitter bio), replace Nan with empty string
X["description"] = X["description"].fillna("")


# Handling categorical/text data

# Description column (twitter bio)
# Getting length (word count) of each description
X["descLen"] = X["description"].str.count('\w+')


# Convert 'created' columns to age
now = pd.Timestamp(datetime.now())
X['created'] = pd.to_datetime(X['created'], format='%m/%d/%y %H:%M')
X['created'] = X['created'].where(X['created'] < now, X['created'] - np.timedelta64(100, 'Y'))
X['created'] = (now - X['created']).astype('<m8[Y]')

# Find the number of hashtags used in description (twitter bio)
X["des_hashtag_count"] = X["description"].str.count("#")

# Has user mentioned provided a link in description(bio)
# lower case for convenience
lower_des = X["description"].str.lower()
bool_list = []
# List of other social media platforms/links and abbreviations of them
lst = ['sc:', "sc ", "snap", "insta", "ig:", "ig ", "fb:", "fb ", "facebook", "http", "https", ".com"]
# Need to check for word rather than letters (using reg expressions)
for i in range(len(lower_des)):
    if (any(sub in lower_des[i] for sub in lst)):
        bool_list.append(1)
    else:
        bool_list.append(0)

X["has_mentioned_other_bio"] = bool_list

# Twitter handle ("name")
# Getting length of each name
X["nameLen"] = X["name"].str.len()

# Should be no need to replace NaN values but do it just in case
X["nameLen"] = X["nameLen"].fillna(0)

# Identify whether default link color is used
# 1 for default color, 0 otherwise
X["uses_default_link_color"] = (X["link_color"] == "0084B4").astype(int)

# Handling a random tweet taken from the profile
# tweet length (word count)
X["tweet_length"] = X["text"].str.count('\w+')

# tagged any other accounts in the tweet
X["num_tagged"] = X["text"].str.count('@')

# number of hashtags in the tweet
X["tweet_hashtags"] = X["text"].str.count('#')

# urls in tweets
lower_tweets = X["text"].str.lower()
X["shared_link"] = X["text"].str.contains('((http:|https:)//[^ \<]*[^ \<\.])')
# double check for shorten urls
X["shortened_urls"] = X["text"].str.contains('https?://t\.co/\S+')
# combine both
X["tweet_has_link"] = X["shared_link"] | X["shortened_urls"]
<<<<<<< HEAD

# Normalizing data
=======
# Note!!!!!!! Later drop shared_links and shortened urls

# tweet length (word count)
X["tweet_length"] = X["text"].str.count('\w+')

>>>>>>> f1e95d49d902e04baf924580d01ac0274bcceda0
X["fav_number"] = normalizeCol("fav_number")
X["descLen"] = normalizeCol("descLen")
X["tweet_count"] = normalizeCol("tweet_count")
X["nameLen"] = normalizeCol("nameLen")
X["tweet_length"] = normalizeCol("tweet_length")
X["created"] = normalizeCol("created")

# Drop categorical data that had been processed
X = X.drop(columns="description")
X = X.drop(columns="link_color")
X = X.drop(columns="name")
X = X.drop(columns="text")
X = X.drop(columns="shortened_urls")
X = X.drop(columns="shared_link")

#Save as csv file
<<<<<<< HEAD
X.to_csv("Processed data.csv", ",")
=======
X.to_csv("ML-1819--task-107--team-32_cleanedUpData.csv", ",")
>>>>>>> f1e95d49d902e04baf924580d01ac0274bcceda0

