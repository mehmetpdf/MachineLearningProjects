# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 12:53:11 2019

@author: mehmet.yilmaz
"""

#%% import library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%% read data
data = pd.read_csv("data.csv")
data.drop(["id", "Unnamed: 32"], axis = 1, inplace = True)

#%% select data and show anyone two features
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

plt.scatter(M.radius_mean, M.texture_mean, color = "red", label = "bad")
plt.scatter(B.radius_mean, B.texture_mean, color = "green", label = "good")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

#%% convert string to int
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

#%% find x and y
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis = 1)

#%% normalization
x = (x_data - np.min(x_data)) / ((np.max(x_data) - np.min(x_data)))

#%% choose test and train data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)

#%% navie bayes algorithm
from sklearn.naive_bayes import GaussianNB
navie = GaussianNB()
navie.fit(x_train, y_train)

#%% test
print("print test score : ", navie.score(x_test, y_test))

