# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 20:51:47 2018

@author: Chi Ho Wong
"""
# Set proper file directory
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
root_path=os.getcwd()
print root_path

import pandas as pd

url = "data_knn.csv"

insider = pd.read_csv(url, header=0)

## Create an 'X' matrix by dropping the irrelevant columns.
X = insider[['type_num', 'size_adv', "size_out_pct", "days_fm_last", "mkt_cap"]]
y = insider.return_30d_num

from sklearn.model_selection import train_test_split
## Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

## Import the Classifier.
from sklearn.neighbors import KNeighborsClassifier
## Instantiate the model with 5 neighbors. 
knn = KNeighborsClassifier(n_neighbors=5)
## Fit the model on the training data.
knn.fit(X_train, y_train)
## See how the model performs on the test data.
print knn.score(X_test, y_test)

