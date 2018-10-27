from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from patsy import dmatrices

df = pd.read_csv("ML-1819--task-107--team-32_cleanedUpData.csv", ',')
print(df)
Y = df.ix[:,1]
X = df.ix[:, 2:-1]
