# -*- coding: utf-8 -*-
"""
ML Twitter Data Preprocessing
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from datetime import date, datetime
import re


def calculate_age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


# Function created to clean up dataset
def CleanUp(dataset):
    # Removing these features because they were not taken from twitter
    dataset = dataset.drop(columns=["_unit_id", "_golden", "_unit_state", "_trusted_judgments", "_last_judgment_at"])
    dataset = dataset.drop(columns=["gender:confidence", "profile_yn", "profile_yn:confidence", "gender_gold"])
    dataset = dataset.drop(columns=["profile_yn_gold", "profileimage"])
    return dataset


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

# Remove tweet location because it is not relevant or quanitifiable
X = X.drop(columns="tweet_location")

# Remove retweet_count because of insufficient data / data seems impossible
X = X.drop(columns="retweet_count")

# Remove user timezone because majority of the data is NaN
X = X.drop(columns="user_timezone")

"""
#Cleaning remaining data
#Remove NaN values from user_timezone
timezones = X.groupby("user_timezone").count()
X["user_timezone"] = X["user_timezone"].fillna("nt")
"""
timezones = X.groupby("sidebar_color").count()
# Removing nan values (setting to 0)
X["description"] = X["description"].fillna("")

# Description column (twitter bio)
# Getting length (word count) of each description
X["descLen"] = X["description"].str.count('\w+')

# Removing nan values (setting to 0)
X["descLen"] = X["descLen"].fillna(0)

# colorS = X.groupby("user_timezone").count()
###################################################################
#############           CLEAN UP END                ###############
###################################################################

# Convert 'created' columns to age
now = pd.Timestamp(datetime.now())

X['created'] = pd.to_datetime(X['created'], format='%m/%d/%y %H:%M')
X['created'] = X['created'].where(X['created'] < now, X['created'] - np.timedelta64(100, 'Y'))
X['created'] = (now - X['created']).astype('<m8[Y]')

# Find the count of hashtags in description
X["des_hashtag_count"] = X["description"].str.count("#")

# Find social media reference is present or not
#X["social_media"] = X["description"].str.contains("@|Instagram")

# Find whether an http link is present or not
#X["http_link"] = X["description"].str.contains("http")

# has mentioned any social in description
# lower case for convenience
lower_des = X["description"].str.lower()
bool_list = []
lst = ['sc', "snap", "insta", "ig", "http", "https", "fb", "facebook", ".com"]
for i in range(len(lower_des)):
    if (any(sub in lower_des[i] for sub in lst)):
        bool_list.append(1)
    else:
        bool_list.append(0)

X["has_mentioned_other_bio"] = bool_list

# Normalize data
from sklearn.preprocessing import MinMaxScaler

favs = X["fav_number"].values.astype(float)
favs = favs.reshape(1, -1)
min_max_scaler = MinMaxScaler()
scaled = min_max_scaler.fit_transform(favs)
normalized = pd.DataFrame(scaled)

# Twitter handle ("name")
# Getting length of each name
X["nameLen"] = X["name"].str.len()

# Should be no need to replace NaN values but do it just in case
X["nameLen"] = X["nameLen"].fillna(0)

# Identify whether default link color is used
# 1 for default color, 0 otherwise
X["uses_default_link_color"] = (X["link_color"] == "0084B4").astype(int)

# tweets
# tagged any other accounts in the
X["num_tagged"] = X["text"].str.count('@')

# number of hashtags
X["tweet_hashtags"] = X["text"].str.count('#')

# urls in tweets
lower_tweets = X["text"].str.lower()
X["shared_link"] = X["text"].str.contains('((http:|https:)//[^ \<]*[^ \<\.])')
# double check for shorten urls
X["shortened_urls"] = X["text"].str.contains('https?://t\.co/\S+')
# combine both
X["tweet_has_link"] = X["shared_link"] | X["shortened_urls"]

# Note!!!!!!! Later drop shared_links and shortened urls

# tweet length (word count)
X["tweet length"] = X["text"].str.count('\w+')

#print(X["tweet length"].iloc[0])

