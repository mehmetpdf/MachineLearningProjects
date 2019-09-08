# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:48:19 2019

@author: mehmet.yilmaz
"""

import pandas as pd
import numpy as np

#%% read data
data = pd.read_csv("data.csv")
data.drop(["id", "Unnamed: 32"], axis = 1, inplace = True)

#%% convert string to int
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

#%% find y and x
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis = 1)

#%% normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

#%% select test and train data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

#%%
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

#%%
print("scrore : ", dt.score(x_test,y_test))