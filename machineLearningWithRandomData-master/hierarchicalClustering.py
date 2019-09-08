# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 16:07:40 2019

@author: mehmet.yilmaz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% create dataset

# 1000 data data üret, ortalamasi 25 olsun.. Ve bu 1000 adet data 20 ile 30 arasinda olsun

# class1
x1 = np.random.normal(25, 5, 100)
y1 = np.random.normal(25, 5, 100)

# class2
x2 = np.random.normal(55, 5, 100)
y2 = np.random.normal(60, 5, 100)

# class3
x3 = np.random.normal(55, 5, 100)
y3 = np.random.normal(15, 5, 100)

x = np.concatenate((x1, x2, x3), axis=0)
y = np.concatenate((y1, y2, y3), axis=0)

dictionary = {"x": x, "y": y}
data = pd.DataFrame(dictionary)

plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.scatter(x3, y3)
plt.show()

# %% dendogram
from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(data, method="ward")
dendrogram(merg, leaf_rotation=90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()

# %%  hierarchical clustering
from sklearn.cluster import AgglomerativeClustering

hg = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
cluster = hg.fit_predict(data)

data["label"] = cluster

# %%

plt.scatter(data.x[data.label == 0], data.y[data.label == 0], color="blue")
plt.scatter(data.x[data.label == 1], data.y[data.label == 1], color="green")
plt.scatter(data.x[data.label == 2], data.y[data.label == 2], color="orange")
plt.show()

