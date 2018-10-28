import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import graphviz

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
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
}


"""""
# creating data
model = XGBClassifier()
#fitting data
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
predictions = [round(value) for value in Y_pred]
# evaluate predictions
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
"""
dtrain = xgb.DMatrix(X_train, label=Y_train)
dtest = xgb.DMatrix(X_test, label= Y_test)
num_rounds = 10
train = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds)


# save model
#train.save_model('001.model')

Y_predict = train.predict(dtest, ntree_limit=train.best_ntree_limit)

predictions = np.where(Y_predict<0.49,0,1)
#print(predictions)
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

xgb.plot_importance(train)
xgb.plot_tree(train, num_trees=4)
xgb.to_graphviz(train, num_trees=4)
plt.show()

