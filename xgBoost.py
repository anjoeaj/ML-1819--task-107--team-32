import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import graphviz
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn import model_selection

# create test and training data
seed = 5
test_size = 0.3

training_data = pd.read_csv("ML-1819--task-107--team-32_cleanedUpDataTrainingSet.csv", ',')
testing_data = pd.read_csv("ML-1819--task-107--team-32_cleanedUpDataTestSet.csv", ",")

X_train = training_data.iloc[:, 2:-1]
Y_train = training_data.iloc[:, 1]
X_test = testing_data.iloc[:, 2:-1]
Y_test = testing_data.iloc[:, 1]


params = {
# logistic model
  'objective' : 'binary:logistic',
# logloss
 'eval_metric' : 'logloss',
# learning rate
  'eta' : 0.1,
    #depth of tree
  "max_depth" : 6,
    # min sum of weights
    # should be high enough to prevent over fitting
    # but not too high for over fitting
    'min_child_weight' : 10,
    # the min loss value require to split
    'gamma' : 0.70,
    # fraction of observations to be included in each tree
    # generally varies from 0.5-1
    'subsample' : 0.75,
    # fraction of column to be randomly sample in each tree
    "colsample_bytree" : 0.95,
    # regularization coefficients
    "alpha" : 2e-05,
    'lambda' : 10
}

params1 = {
# logistic model
  'objective' : ['binary:logistic'],

    #depth of tree
  "max_depth" : [4, 5, 6],
    # min sum of weights
    # should be high enough to prevent over fitting
    # but not too high for over fitting
    'min_child_weight' : [8,10,11],
    # the min loss value require to split
    'gamma' : [0.5, 0.7,0.8, 1.2],
    # fraction of observations to be included in each tree
    # generally varies from 0.5-1
    'subsample' : [0.55, 0.75, 0.95],
    # fraction of column to be randomly sample in each tree
    "colsample_bytree" : [0.6,0.8,0.95],
    # regularization coefficients
    #"alpha" : [0.00002],
    #'lambda' : [10]
}

dtrain = xgb.DMatrix(X_train, label=Y_train)
dtest = xgb.DMatrix(X_test, label= Y_test)
num_rounds = 10
train = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds)


# save model
train.save_model('001.model')

# yo!!! just finding the best params
folds = 3
param_comb = 5

clf = xgb.XGBClassifier(learning_rate=0.02, n_estimators=600,
                    silent=True, nthread=1)
gridsearch = model_selection.GridSearchCV(clf,params1)
gridsearch.fit(X_train, Y_train)
#skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)
#random_search = RandomizedSearchCV(clf, params1, n_iter=param_comb, scoring='roc_auc', n_jobs=4,
#                                   cv=skf.split(X_train,Y_train), verbose=3, random_state=1001 )
print(random_search)
#random_search.fit(X=X_train, y=Y_train)


Y_predict = train.predict(dtest, ntree_limit=train.best_ntree_limit)

predictions = np.where(Y_predict<0.49,0,1)
#print(predictions)
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

xgb.plot_tree(train, num_trees=4)
xgb.to_graphviz(train, num_trees=4)
xgb.plot_importance(train)
plt.show()

