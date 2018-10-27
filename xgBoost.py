import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("ML-1819--task-107--team-32_cleanedUpData.csv", ',')

# split in X and Y
Y = df.iloc[:, 1]
X = df.iloc[:, 2:-1]

# create test and training data
seed = 5
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# creating data
model = XGBClassifier()
#fitting data
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
