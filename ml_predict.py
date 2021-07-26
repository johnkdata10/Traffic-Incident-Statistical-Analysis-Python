# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 13:50:40 2021

@author: JohnKramarczyk
"""

# !pip install statsmodels

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from pandas import read_csv
import re
import xlrd
import openpyxl
from openpyxl import Workbook
import xlsxwriter

import sklearn
import random
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics

from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import warnings
warnings.filterwarnings('ignore')

# %matplotlib inline

# plt.rcParams['figure.figsize'] = [20.0, 7.0]
# plt.rcParams.update({'font.size': 22,})

# sns.set_palette('viridis')
# sns.set_style('white')
# sns.set_context('talk', font_scale=0.8)

onehot_df = pd.read_excel('C:/Users/JohnKramarczyk/Documents/__ML Project/TestSET.xlsx')
# onehot_df.corr()

X = onehot_df.iloc[:, 1:-1]
X.head()

Y = onehot_df.iloc[:, -1]
Y.head()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

list(onehot_df.columns)

model = LinearRegression()
model.fit(X_train,y_train)

print(model.intercept_)

coeff_parameter = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
coeff_parameter

predictions = model.predict(X_test)
predictions

sns.regplot(y_test,predictions)

X_train_Sm= sm.add_constant(X_train)
X_train_Sm= sm.add_constant(X_train)
ls=sm.OLS(y_train,X_train_Sm).fit()
print(ls.summary())

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# prepare for modeling
X_train = onehot_df.drop(['Total_Incidents'], axis=1)
y_train = onehot_df['Total_Incidents']

X_test = onehot_df.drop(['employee_id'], axis=1)

# scaling data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# function to get cross validation scores
def get_cv_scores(model):
    scores = cross_val_score(model,
                             X_train,
                             y_train,
                             cv=5,
                             scoring='r2')
    
    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))
    print('\n')

lr = LinearRegression().fit(X_train, y_train)
get_cv_scores(lr)

# function to get cross validation scores
from sklearn.linear_model import Ridge
# Train model with default alpha=1
ridge = Ridge(alpha=1).fit(X_train, y_train)
# get cross val scores
get_cv_scores(ridge)

# find optimal alpha with grid search
alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
param_grid = dict(alpha=alpha)
grid = GridSearchCV(estimator=ridge, param_grid=param_grid, scoring='r2', verbose=1, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)
print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)

logistic = linear_model.LogisticRegression(C=1, class_weight={1:0.6, 0:0.4}, penalty='l1', solver='liblinear')
get_cv_scores(logistic)

predictions1 = logistic.fit(X_train, y_train).predict_proba(X_test)
onehot_df['predictions1'] = predictions1
len(predictions)
len(onehot_df)
onehot_df.head()

# define models
ridge = linear_model.Ridge()
lasso = linear_model.Lasso()
elastic = linear_model.ElasticNet()
lasso_lars = linear_model.LassoLars()
bayesian_ridge = linear_model.BayesianRidge()
logistic = linear_model.LogisticRegression(solver='liblinear')
sgd = linear_model.SGDClassifier()

models = [ridge, lasso, elastic, lasso_lars, bayesian_ridge, logistic, sgd]

# loop through list of models
for model in models:
    print(model)
    get_cv_scores(model)

penalty = ['l1', 'l2']
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]
solver = ['liblinear', 'saga']

param_grid = dict(penalty=penalty,
                  C=C,
                  class_weight=class_weight,
                  solver=solver)

grid = GridSearchCV(estimator=logistic, param_grid=param_grid, scoring='roc_auc', verbose=1, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)

logistic = linear_model.LogisticRegression(C=1, class_weight={1:0.6, 0:0.4}, penalty='l1', solver='liblinear')
get_cv_scores(logistic)

predictions2 = logistic.fit(X_train, y_train).predict_proba(X_test)

onehot_df['predictions2'] = predictions2
len(predictions)
len(onehot_df)
onehot_df.head()

loss = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
penalty = ['l1', 'l2', 'elasticnet']
alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive']
class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]
eta0 = [1, 10, 100]

param_distributions = dict(loss=loss,
                           penalty=penalty,
                           alpha=alpha,
                           learning_rate=learning_rate,
                           class_weight=class_weight,
                           eta0=eta0)

random = RandomizedSearchCV(estimator=sgd, param_distributions=param_distributions, scoring='roc_auc', verbose=1, n_jobs=-1, n_iter=1000)
random_result = random.fit(X_train, y_train)

print('Best Score: ', random_result.best_score_)
print('Best Params: ', random_result.best_params_)

sgd = linear_model.SGDClassifier(alpha=0.1,
                                 class_weight={1:0.7, 0:0.3},
                                 eta0=100,
                                 learning_rate='optimal',
                                 loss='log',
                                 penalty='elasticnet')
get_cv_scores(sgd)

predictions3 = sgd.fit(X_train, y_train).predict_proba(X_test)
onehot_df['predictions3'] = predictions3
len(predictions)
len(onehot_df)
onehot_df.head()
onehot_df.tail()