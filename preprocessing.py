# -*- coding: utf-8 -*-
"""
ML Twitter Data Preprocessing
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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