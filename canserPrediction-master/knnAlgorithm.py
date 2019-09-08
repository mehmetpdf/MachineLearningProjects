# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 20:22:05 2019

@author: mehmet.yilmaz
"""

# import data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# data read
data = pd.read_csv("data.csv")
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)

# find iyi huylu and kotu huylu data's
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

# look at any two feature
plt.scatter(M.radius_mean, M.texture_mean, color="red", label="kotu")
plt.scatter(B.radius_mean, B.texture_mean, color="green", label="iyi")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()

# find x and y
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)

# normalizastion
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

# get test and train datas
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# knn model
from sklearn.neighbors import KNeighborsClassifier

k = 3;

knn = KNeighborsClassifier(n_neighbors=k)  # n_neighbors = k in knn
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)

# print score
print("{} nn Score : {}".format(k, knn.score(x_test, y_test)))

# find best k in knn so what is the best k?
score_list = []
for each in range(1, 15):
    knn2 = KNeighborsClassifier(n_neighbors=each)  # n_neighbors = k in knn
    knn2.fit(x_train, y_train)
    score_list.append(knn2.score(x_test, y_test))

plt.plot(range(1, 15), score_list)
plt.xlabel("k values")
plt.ylabel("score")
plt.show()


