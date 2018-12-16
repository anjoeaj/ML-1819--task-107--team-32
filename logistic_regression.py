from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score

# NOW USING A TRAIN DATA SET AND TEST DATA SET

#Read twitter data 
dta = pd.read_csv("words_dataset.csv", ',')

#Split columns to X and Y
Y = dta.iloc[:,1]
X = dta.iloc[:, 2:-1]

#sReformat data
Y, X = dmatrices('gender ~ text', dta, return_type="dataframe")

Y = np.ravel(Y)


# instantiate a logistic regression model, and fit with training data for X and y
model = LogisticRegression()
model = model.fit(X, Y)

# check the accuracy on the training set
print(model.score(X, Y))

print(Y.mean())

# Test model on test data
Y_pred = model.predict(X)
print(model.score(X, Y))
