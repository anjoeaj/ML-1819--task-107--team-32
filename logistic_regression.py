from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score

# NOW USING A TRAIN DATA SET AND TEST DATA SET

#Read twitter data 
dta = pd.read_csv("ML-1819--task-107--team-32_cleanedUpData.csv", ',')

#Split columns to X and Y
Y_train = dta.iloc[:,1]
X_train = dta.iloc[:, 2:-1]

#sReformat data
Y_train, X_train = dmatrices('gender ~ created + fav_number + tweet_count + descLen + des_hashtag_count + nameLen + tweet_length + '
                'num_tagged + tweet_hashtags + C(has_mentioned_other_bio) + C(uses_default_link_color) + '
                'C(tweet_has_link)', dta, return_type="dataframe")

Y_train = np.ravel(Y)


# instantiate a logistic regression model, and fit with training data for X and y
model = LogisticRegression()
model = model.fit(X_train, Y_train)

# check the accuracy on the training set
print(model.score(X_train, Y_train))

print(Y_train.mean())

# Test model on test data
#Y_pred = model.predict(X_test)
#print(model.score(X_test, Y_test))
