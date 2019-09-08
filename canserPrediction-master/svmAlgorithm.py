# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:36:42 2019

@author: mehmet.yilmaz
"""

# %% import data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %% read data
data = pd.read_csv("data.csv")
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

# %% selection M and B
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

# %% show anyone features in graphic
plt.scatter(M.radius_mean, M.texture_mean, color="red", label="bad")
plt.scatter(B.radius_mean, B.texture_mean, color="green", label="good")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

# %%
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)

# %% normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

# %% select test and traind data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# %% SVM

from sklearn.svm import SVC

svm = SVC(random_state=1)
svm.fit(x_train, y_train)

# %% test
print("print accuarcy of svm algo : ", svm.score(x_test, y_test))

















