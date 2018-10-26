# -*- coding: utf-8 -*-
"""
ML Twitter Data Preprocessing
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from datetime import date, datetime



def calculate_age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

#Function created to clean up dataset
def CleanUp(dataset):
    # Removing these features because they were not taken from twitter
    dataset = dataset.drop(columns = ["_unit_id", "_golden", "_unit_state", "_trusted_judgments", "_last_judgment_at"])
    dataset = dataset.drop(columns = ["gender:confidence", "profile_yn", "profile_yn:confidence", "gender_gold"])
    dataset = dataset.drop(columns = ["profile_yn_gold", "profileimage"])
    return dataset
        

# Importing the dataset
dataset = pd.read_csv('twitter.csv', encoding = "ISO-8859-1")
# Cleaning the dataset (creators of dataset added features for their specific model)
dataset = CleanUp(dataset)

# y will be the gender of the account
y = dataset.iloc[:, 0].values

# x will be our modifiers
X = dataset.drop(columns = "gender")

#Removing 'tweet coords' because majority are nan values
X = X.drop(columns = "tweet_coord")

#Removing 'tweet ID' because values are random
X = X.drop(columns = "tweet_id")

#Removing 'tweet time' because all tweets were captured within a two minute span
X = X.drop(columns = "tweet_created")

#Remove tweet location because it is not relevant or quanitifiable
X = X.drop(columns = "tweet_location")

#Remove retweet_count because of insufficient data / data seems impossible
X = X.drop(columns = "retweet_count")

#Remove user timezone because majority of the data is NaN
X = X.drop(columns = "user_timezone")

"""
#Cleaning remaining data
#Remove NaN values from user_timezone
timezones = X.groupby("user_timezone").count()
X["user_timezone"] = X["user_timezone"].fillna("nt")
"""

#Description column (twitter bio)
#Getting length of each description
X["descLen"] = X["description"].str.len()

#Removing nan values (setting to 0)
X["descLen"] = X["descLen"].fillna(0)

#Twitter handle ("name")
#Getting length of each name
X["nameLen"] = X["name"].str.len()

#Should be no need to replace NaN values but do it just in case
X["nameLen"] = X["nameLen"].fillna(0)

#colorS = X.groupby("user_timezone").count()
###################################################################
#           CLEAN UP END
###################################################################

#Convert 'created' columns to age
now = pd.Timestamp(datetime.now())

X['created'] = pd.to_datetime(X['created'], format='%m/%d/%y %H:%M')    # 1
X['created'] = X['created'].where(X['created'] < now, X['created'] -  np.timedelta64(100, 'Y'))   # 2
X['created'] = (now - X['created']).astype('<m8[Y]') 




