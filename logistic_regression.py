from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from patsy import dmatrices

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import model_selection

# NOW USING A TRAIN DATA SET AND TEST DATA SET

#Read twitter data 
dta = pd.read_csv("ML-1819--task-107--team-32_cleanedUpDataTrainingSet.csv", ',')

#Split columns to X and Y
Y_train = dta.iloc[:,1]
X_train = dta.iloc[:, 2:-1]

#sReformat data
Y_train, X_train = dmatrices('gender ~ created + fav_number + tweet_count + descLen + des_hashtag_count + nameLen + tweet_length + '
                'num_tagged + tweet_hashtags + C(has_mentioned_other_bio) + C(uses_default_link_color) + '
                'C(tweet_has_link)', dta, return_type="dataframe")

Y_train = np.ravel(Y_train)


#Setup polynomial features
poly = PolynomialFeatures(2)
X_train = poly.fit_transform(X_train)


#-------------------SETUP TEST DATA -------------------------------

dta = pd.read_csv("ML-1819--task-107--team-32_cleanedUpDataTestSet.csv", ',')
Y_test = dta.iloc[:,1]
X_test = dta.iloc[:, 2:-1]
Y_test, X_test = dmatrices('gender ~ created + fav_number + tweet_count + descLen + des_hashtag_count + nameLen + tweet_length + '
                    'num_tagged + tweet_hashtags + C(has_mentioned_other_bio) + C(uses_default_link_color) + '
                    'C(tweet_has_link)', dta, return_type="dataframe")
    
#test data of prediction value to 1D array
Y_test = np.ravel(Y_test)

#use the same polynomial features and 
X_test = poly.fit_transform(X_test)    

parameter_list = [{'penalty':['l1'], 'C': [ 0.01, 0.1, 1, 10, 100, 1000]},
                   {'penalty':['l2'], 'C': [ 0.01, 0.1, 1, 10, 100, 1000]}]


###### TEST FOR MULTIPLE VALUES OF C ########
#values for C added in an array
#c_values = [ 0.01, 0.1, 1, 10, 100, 1000];
#for k in c_values:
# instantiate a logistic regression model, and fit with training data for X and y
#model = LogisticRegression(penalty = 'l2', C = k)
######################################

model = LogisticRegression()

gridsearch = model_selection.GridSearchCV(model, parameter_list)
gridsearch.fit(X_train, Y_train)
print(gridsearch.best_params_)
print(gridsearch.score)

#Fit the model with the transformed data
model = model.fit(X_train, Y_train)

# check the accuracy on the training set
print(model.score(X_train, Y_train)) 

#check mean of Y data                                                                      
#print(Y_train.mean())

# Test model on test data
y_pred = model.predict(X_test)
print(accuracy_score(Y_test, y_pred))
print("---------------------------------------------------------------------") 



