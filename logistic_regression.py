# -*- coding: utf-8 -*-
"""
ML Twitter Data - Algorithms - Logistic Regression

"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


dataset = pd.read_csv('ML-1819--task-107--team-32_cleanedUpData.csv', encoding="ISO-8859-1")

X=dataset[:,0]