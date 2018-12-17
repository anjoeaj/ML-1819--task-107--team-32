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
#model = model.fit(X_train, y_train)
y_score = model.fit(X_train, y_train).decision_function(X_test)

# check the accuracy on the training set
print(model.score(X_train, y_train))

print(y_train.mean())

# Test model on test data
Y_pred = model.predict(X_test)
print(model.score(X_test, y_test))

from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 1]).ravel()
(tn, fp, fn, tp)
(0, 2, 1, 1)




# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()



gender_arr = np.array(y_test['gender'])
#gender_arr.ravel()
fpr, tpr, _ = roc_curve(gender_arr, y_score)
roc_auc = auc(fpr, tpr)
#for i in range(n_classes):
#    fpr[i], tpr[i], _ = roc_curve(gender_arr, y_score)
#    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr, tpr, _ = roc_curve(gender_arr.ravel(), y_score.ravel())
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()