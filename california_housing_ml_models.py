# -*- coding: utf-8 -*-
"""California-Housing-ML-Models.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1g2kuzddMsqNSfOaOTI4W71WpDKGslGAE
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error

### Regression Problem - Predict how the blood sugar level will be after 1 year!
dataset = fetch_california_housing()
print(dataset.DESCR)

print(dataset.keys())

x = dataset['data']
y = dataset['target']
feature_names = dataset['feature_names']

from sklearn.feature_selection import mutual_info_regression, SelectPercentile, SelectKBest
mi = mutual_info_regression(x, y)

x.shape

print(mi)

# Visualise Feature Selection
plt.figure(figsize=(10, 6))
plt.bar(feature_names, mi)
plt.show()

# Option 1
x_new = SelectPercentile(mutual_info_regression, percentile=50).fit_transform(x, y)
print(x_new.shape)

# Option 2
x_new = SelectKBest(mutual_info_regression, k=5).fit_transform(x, y)
print(x_new.shape)

x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.2)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(r2_score(y_test, y_pred))
mse_err = mean_squared_error(y_test, y_pred)
print(mse_err)

"""***Pearson Correlation***"""

dataset = fetch_california_housing()
x = dataset['data']
y = dataset['target']
feature_names = dataset['feature_names']
print(x.shape)

"""Two ways of usage

a. Analyse the relationship between 'each' individual feature (input) and output
"""

from sklearn.feature_selection import f_regression, SelectKBest

x_new = SelectKBest(f_regression, k=8).fit_transform(x, y)
print(x_new.shape)

x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.2)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(r2_score(y_test, y_pred))
mse_err = mean_squared_error(y_test, y_pred)
print(mse_err)

"""b. Analyses relationship b/w the features themselves

Core Idea: Drop features which are highly correlated (dealing only on the input side)
"""

import pandas as pd

x_pd = pd.DataFrame(x, columns=feature_names)
x_pd.head(10)

x_pd.corr()

import seaborn as sns

#correlation map
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(x_pd.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

x_new = x_pd.drop(['Latitude', 'Longitude'], axis=1)
x_new.head()

x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.2)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(r2_score(y_test, y_pred))
mse_err = mean_squared_error(y_test, y_pred)
print(mse_err)

"""***Recurive Feature Elimination (RFE)***
Given an estimator that assigns weights/coeffecients to the features (eg: linear model),
It starts out by training the model on all the features.
Then, recursively, removes the least important features, and re-trains the model.
This process is repeated until we have the desired number of features.
"""

from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso

# Regression
california = fetch_california_housing()
X, y = california.data, california.target

estimator = Lasso()
selector = RFE(estimator, n_features_to_select=5, step=1).fit(X, y)
print(selector.ranking_, )

X_new = selector.transform(X)
print(X_new.shape)

x_train, x_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(r2_score(y_test, y_pred))
mse_err = mean_squared_error(y_test, y_pred)
print(mse_err)

"""**Sequential Feature Selection**
Doesn't require the underlying model to provide co-efficient weights, such as SelectFromModel, and RFE

Forward: Finds the best new feature to add to the set of selected features. Concretely, we initially start with zero features and find the one feature that maximizes a cross-validated score when an estimator is trained on this single feature. Once that first feature is selected, we repeat the procedure by adding a new feature to the set of selected features. The procedure stops when the desired number of selected features is reached, as determined by the n_features_to_select parameter.

Backward: Starts from 'n' features, and removes 1 at a time till we reach the desired number of features
"""

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeCV

# Regression
california = fetch_california_housing()
X, y = california.data, california.target
ridge = RidgeCV().fit(X, y)

sfs_forward = SequentialFeatureSelector(
    ridge, n_features_to_select="auto", direction="forward"
).fit(X, y)

print(sfs_forward)

X_new = sfs_forward.transform(X)
X_new.shape

## Important: You can now use any model with these newly selected features
x_train, x_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(r2_score(y_test, y_pred))
mse_err = mean_squared_error(y_test, y_pred)
print(mse_err)

"""**Unsupervised**"""

#PCA
dataset = fetch_california_housing()
x = dataset['data']
y = dataset['target']
feature_names = dataset['feature_names']

print(x.shape)

from sklearn.decomposition import PCA
X_new = PCA(n_components=4, svd_solver='full').fit_transform(x)
print(X_new.shape)

x_train, x_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(r2_score(y_test, y_pred))
mse_err = mean_squared_error(y_test, y_pred)
print(mse_err)

