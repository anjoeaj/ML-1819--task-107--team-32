from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import seaborn as sns # For barcharts (not used right now)

#nltk.download()

lemmatizer = WordNetLemmatizer()

# NOW USING A TRAIN DATA SET AND TEST DATA SET

#Read twitter data 
dta = pd.read_csv("words_dataset.csv", ',')

#bow_vectorizer2 = CountVectorizer(max_df=0.90, min_df=2, max_features=200, stop_words='english')
#bow_hashtags = bow_vectorizer2.fit_transform(dta["text"])
#for i in range(len(dta)):
#    dta["text"][i] = bow_hashtags[i].toarray()

#Split columns to X and Y
Y = dta.iloc[:,1]
X = dta.iloc[:, 2:-1]

#sReformat data
Y, X = dmatrices('gender ~ created + fav_number + tweet_count + descLen + nameLen', dta, return_type="dataframe")

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

from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 1]).ravel()
(tn, fp, fn, tp)
(0, 2, 1, 1)
