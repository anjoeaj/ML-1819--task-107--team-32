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


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
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
n_classes = Y.shape[0]
#sReformat data
Y, X = dmatrices('gender ~ created + fav_number + tweet_count + descLen + nameLen', dta, return_type="dataframe")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3,
                                                    random_state=0)
Y = np.ravel(Y)


# instantiate a logistic regression model, and fit with training data for X and y
#model = LogisticRegression()
model = OneVsRestClassifier(LogisticRegression())
model = model.fit(X_train, y_train)
#y_score = model.fit(X_train, y_train).decision_function(X_test)

# check the accuracy on the training set
print(model.score(X_train, y_train))

print(y_train.mean())

# Test model on test data
Y_pred = model.predict(X_test)
print(model.score(X_test, y_test))

######################### PLOT ROC CURVE ###############################

#Get the true positives and false positives 
probs = model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# draw the graph 
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'darkorange', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--',color='navy')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

######################### ROC CURVE END ###############################




from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 1]).ravel()
(tn, fp, fn, tp)
(0, 2, 1, 1)