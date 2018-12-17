import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from sklearn.model_selection import train_test_split

df = pd.read_csv("words_dataset.csv")

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['gender'],
                                                    test_size=0.25)

# setting up count vectorizer
vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=2)
vectorized_X = vectorizer.fit_transform(X_train)

print(vectorized_X.toarray())

vectorized_X_test = vectorizer.fit_transform(X_test)

params = {
    "objective": "reg:logistic",

    # logless for cross-validation
    "eval_metric": "auc",

    # learning rate
    "learning_rate": 0.1,

    # depth of tree
    "max_depth": 6,

    # min sum of weights
    # should be high enough to prevent over fitting
    # but not too high for over fitting
    "min_child_weight": 10,

    # the min loss value require to split
    "gamma": 0.70,

    # fraction of observations to be included in each tree
    # generally varies from 0.5-1
    "subsample": 0.75,

    # fraction of column to be randomly sample in each tree
    "colsample_bytree": 0.95,

    # regularization coefficients
    "alpha": 2e-05,
    "lambda": 10
}

data_dmatrix = xgb.DMatrix(data=vectorized_X, label=y_train.values)

booster = xgb.train(params, data_dmatrix)

