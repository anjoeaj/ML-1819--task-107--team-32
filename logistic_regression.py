# -*- coding: utf-8 -*-
"""
Logistic Regression
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection

from sklearn import metrics
from sklearn.metrics import roc_curve, auc

#Import statistical features
dataset = pd.read_csv("features_3000words.csv", ',')

# Separate Y column
gender = dataset['gender']
y_train, y_test = train_test_split(gender, test_size = 0.25, random_state = 0)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# Separate X values
features = dataset.drop(columns='gender')
X_train, X_test = train_test_split(features, test_size = 0.25, random_state = 0)

# GridSearch
gridModel = LogisticRegression()
trim_parameter_listL1 = [{'C': [0.1, 0.2, 0.3, 0.4, 0.5, 1], "penalty":["l1", "l2"], 'solver' : ['liblinear'], 'max_iter': [100]}]
trim_parameter_listL2 = [{'C': [0.01, 0.1, 1, 10, 100, 1000], "penalty":["l2"], 'solver' : ['newton-cg','lbfgs','sag'], 'max_iter': [500]}]

gridsearch = model_selection.GridSearchCV(gridModel, trim_parameter_listL1)
gridsearch.fit(X_train, y_train)
print(gridsearch.best_params_)
print(gridsearch.score)

# Model_best params
model = LogisticRegression(C = 0.2, penalty = "l2", solver = 'liblinear' , max_iter = 100)
model.fit(X_train, y_train) # training the model
print(model.score(X_train, y_train))
predictions = model.predict(X_test)
score = model.score(X_test, y_test)
print(score)

"""from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(cv=3,Cs=2, random_state=0,multi_class='ovr', solver='liblinear', penalty='l2', max_iter= 500).fit(X_train, y_train)
print(clf.score(X_train, y_train))
Y_pred = clf.predict(X_test)
print(clf.score(X_test, y_test))"""

########################## SVM TEST #############################
#{'kernel':['rbf'], 'gamma': ['auto', 0.001, 0.01, 0.1, 0, 1, 10, 100, 1000], 'C': [0.01, 0.1, 1, 10, 100, 1000]},	
#{'kernel':['sigmoid'], 'gamma': ['auto', 0.001, 0.01, 0.1, 0, 1, 10, 100, 1000], 'C': [ 0.01, 0.1, 1, 10, 100, 1000]},	
#{'kernel':['poly'], 'gamma': ['auto', 0.001, 0.01, 0.1, 0, 1, 10, 100, 1000], 'C': [ 0.01, 0.1, 1, 10, 100, 1000], 'degree': [2, 3]}]	
from sklearn import svm
modelSVM = svm.SVC()	
trim_parameter_list = [{'kernel':['sigmoid'], 'gamma': ['auto', 0.1, 1, 10, 100, 1000], 'C': [0.1, 1, 10, 1000]}]	
gridsearch = model_selection.GridSearchCV(modelSVM, trim_parameter_list)	
gridsearch.fit(X_train, y_train)	
print(gridsearch.best_params_)	
print(gridsearch.score)

model = svm.SVC()
model.fit(X_train, y_train) # training the model
print(model.score(X_train, y_train))
Y_pred = model.predict(X_test)
print(model.score(X_test, y_test))

######################### PLOT ROC CURVE ###############################

# Plot the graph for test data
probs = model.predict_proba(X_train)
preds = probs[:,1]
fpr1, tpr1, threshold = metrics.roc_curve(y_train, preds)
roc_auc1 = metrics.auc(fpr1, tpr1)

#Get the true positives and false positives 
probs = model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# draw the graph 
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')

#plot test data ROC curve
plt.plot(fpr, tpr, 'darkorange', label = 'Test AUC = %0.2f' % roc_auc)

#plot training data ROC curve
plt.plot(fpr1, tpr1, 'g', label = 'Train AUC = %0.2f' % roc_auc1)

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--',color='navy')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

######################### ROC CURVE END ###############################

######################### CONFUSION MATRIX ############################
import seaborn as sns
# Convert predictions from 0,1 back to female, male
y_test_string = ["" for x in range(len(y_test))]
predictions_string = ["" for x in range(len(predictions))]
for i in range(len(y_test)):
    if(y_test[i]==0):
        y_test_string[i] = "Female"
    else:
        y_test_string[i] = "Male"
        
    if(predictions[i]==0):
        predictions_string[i] = "Female"
    else:
        predictions_string[i] = "Male"
        
gender_labels = ["Female","Male"]
confMat = metrics.confusion_matrix(y_test_string, predictions_string)
print(confMat)
plt.figure(figsize=(12,12))
sns.heatmap(confMat, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
tick_position = [0.5, 1.5]
plt.xticks(tick_position, gender_labels)
plt.yticks(tick_position, gender_labels, va = 'center')
plt.ylabel('Ground Truth');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {:.3f}'.format(score)
plt.title(all_sample_title, size = 15);