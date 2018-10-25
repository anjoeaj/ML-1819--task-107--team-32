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
    dataset = dataset.drop(columns = ["_unit_id", "_golden", "_unit_state", "_trusted_judgments", "_last_judgment_at"])
    dataset = dataset.drop(columns = ["gender:confidence", "profile_yn", "profile_yn:confidence", "gender_gold"])
    dataset = dataset.drop(columns = ["profile_yn_gold", "profileimage"])
    
    return dataset
        

# Importing the dataset
dataset = pd.read_csv('twitter.csv', encoding = "ISO-8859-1")
# Cleaning the dataset (creators of dataset added features for their specific model)
dataset = CleanUp(dataset)



