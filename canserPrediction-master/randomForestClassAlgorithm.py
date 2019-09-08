# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:23:48 2019

@author: mehmet.yilmaz
"""

#%% import library
import pandas as pd
import numpy as np

#%% read data
data = pd.read_csv("data.csv")
data.drop(["id", "Unnamed: 32"], axis = 1, inplace = True)
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

#%% find x and y
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis = 1)

#%% normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

#%% selection test and train data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

#%% random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state = 1)
rf.fit(x_train, y_train)

#%% score
print("score : ", rf.score(x_test, y_test))