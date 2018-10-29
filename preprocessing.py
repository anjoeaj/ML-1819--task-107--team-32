# -*- coding: utf-8 -*-
"""
ML Twitter Data Preprocessing
"""

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import date, datetime

# Determine from date of join, how many years old each account is
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
    colVals = dataset[colName].values.astype(float)
    norm = colVals/np.linalg.norm(colVals)
    return norm


# Importing the dataset
dataset = pd.read_csv('twitter.csv', encoding="ISO-8859-1")

# Cleaning the dataset (creators of dataset added features for their specific model)
dataset = CleanUp(dataset)

# Gender data contains: male, female, unknown, brand
# Remove brand and unknown rows
valid = ["male", "female"]
dataset = dataset[dataset['gender'].isin(valid)]
dataset = dataset.reset_index(drop=True)

# Removing 'tweet coords' because majority are nan values
dataset = dataset.drop(columns="tweet_coord")

# Removing 'tweet ID' because values are random
dataset = dataset.drop(columns="tweet_id")

# Removing 'tweet time' because all tweets were captured within a two minute span
dataset = dataset.drop(columns="tweet_created")

# Removing tweet location because it is not relevant or quanitifiable
dataset = dataset.drop(columns="tweet_location")

# Removing retweet_count because of insufficient data / data seems implausible
dataset = dataset.drop(columns="retweet_count")

# Removing user timezone because majority of the data is NaN
dataset = dataset.drop(columns="user_timezone")

# Removing sidebar color because default is unknown and feature is deprecated
dataset = dataset.drop(columns="sidebar_color")


# In description columns (twitter bio), replace Nan with empty string
dataset["description"] = dataset["description"].fillna("")


# Handling categorical/text data
# Description column (twitter bio)
# Getting length (word count) of each description
dataset["descLen"] = dataset["description"].str.count('\w+')

# Convert 'created' columns to age
now = pd.Timestamp(datetime.now())
dataset['created'] = pd.to_datetime(dataset['created'], format='%m/%d/%y %H:%M')
dataset['created'] = dataset['created'].where(dataset['created'] < now, dataset['created'] - np.timedelta64(100, 'Y'))
dataset['created'] = (now - dataset['created']).astype('<m8[Y]')

# Find the number of hashtags used in description (twitter bio)
dataset["des_hashtag_count"] = dataset["description"].str.count("#")

# Has user mentioned provided a link in description(bio)
# lower case for convenience
lower_des = dataset["description"].str.lower()
bool_list = []
# List of other social media platforms/links and abbreviations of them
lst = ['sc:', "sc ", "snap", "insta", "ig:", "ig ", "fb:", "fb ", "facebook", "http", "https", ".com"]
# Need to check for word rather than letters (using reg expressions)
for i in range(len(lower_des)):
    if (any(sub in lower_des[i] for sub in lst)):
        bool_list.append(1)
    else:
        bool_list.append(0)

dataset["has_mentioned_other_bio"] = bool_list

# Twitter handle ("name")
# Getting length of each name
dataset["nameLen"] = dataset["name"].str.len()

# Should be no need to replace NaN values but do it just in case
dataset["nameLen"] = dataset["nameLen"].fillna(0)

# Identify whether default link color is used
# 1 for default color, 0 otherwise
dataset["uses_default_link_color"] = (dataset["link_color"] == "0084B4").astype(int)

# Handling a random tweet taken from the profile
# tweet length (word count)
dataset["tweet_length"] = dataset["text"].str.count('\w+')

# tagged any other accounts in the tweet
dataset["num_tagged"] = dataset["text"].str.count('@')

# number of hashtags in the tweet
dataset["tweet_hashtags"] = dataset["text"].str.count('#')

# urls in tweets
lower_tweets = dataset["text"].str.lower()
dataset["shared_link"] = dataset["text"].str.contains('((http:|https:)//[^ \<]*[^ \<\.])')
# double check for shorten urls
dataset["shortened_urls"] = dataset["text"].str.contains('https?://t\.co/\S+')
# combine both
dataset["tweet_has_link"] = dataset["shared_link"] | dataset["shortened_urls"]

# Normalizing data
dataset["fav_number"] = normalizeCol("fav_number")
dataset["descLen"] = normalizeCol("descLen")
dataset["tweet_count"] = normalizeCol("tweet_count")
dataset["nameLen"] = normalizeCol("nameLen")
dataset["tweet_length"] = normalizeCol("tweet_length")
dataset["created"] = normalizeCol("created")

# Convert remaining categorical data 
# 'True', 'False', to 1, 0
dataset.tweet_has_link = dataset.tweet_has_link.astype(int)

# 'Male, 'Female' to 1, 0
dataset["gender"] = np.where(dataset["gender"] == "male", 1, 0)

# Drop categorical data that has been processed
dataset = dataset.drop(columns="description")
dataset = dataset.drop(columns="link_color")
dataset = dataset.drop(columns="name")
dataset = dataset.drop(columns="text")
dataset = dataset.drop(columns="shortened_urls")
dataset = dataset.drop(columns="shared_link")

#Split data into training and test sets
train, test = train_test_split(dataset, test_size = 0.25, random_state = 0)

#Save as csv file
train.to_csv("ML-1819--task-107--team-32_cleanedUpDataTrainingSet.csv", ",")
test.to_csv("ML-1819--task-107--team-32_cleanedUpDataTestSet.csv", ",")