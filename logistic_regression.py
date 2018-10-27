from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from patsy import dmatrices

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score

#Read twitter data 
dta = pd.read_csv("ML-1819--task-107--team-32_cleanedUpData.csv", ',')

#Split columns to X and Y
Y = dta.ix[:,1]
X = dta.ix[:, 2:-1]

# Reformat data
Y, X = dmatrices('gender ~ created + fav_number + tweet_count + descLen + des_hashtag_count + nameLen + tweet_length + '
                'num_tagged + tweet_hashtags + C(has_mentioned_other_bio) + C(uses_default_link_color) + '
                'C(tweet_has_link)', dta, return_type="dataframe")

Y = np.ravel(Y)


# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(X, Y)

# check the accuracy on the training set
print(model.score(X, Y))

print(Y.mean())
