# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 07:45:44 2019

@author: mehmet.yilmaz
"""

# %% info
# 1- read Data
# 2- define y ve x
# 3- make normalizasyon
# 4- define test ve train data
# 5- set initalize values
# 6- define sigmoid function
# 7- define forward and backward function
# 8- update method for w and b
# 9- need to predict for test_data

# %% import library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% read data

data = pd.read_csv("voice.csv")
data.label = [1 if each == "male" else 0 for each in data.label]  # 0:female, 1:male
print(data.info())

y = data.label.values
x_data = data.drop(["label"], axis=1)

# normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

# test and train data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train : ", x_train.shape)
print("x_test : ", x_test.shape)
print("y_train : ", y_train.shape)
print("y_test : ", y_test.shape)


# %% initalize
def initalize_weight_and_bias(dimension):
    w = np.full((dimension, 1), 0.01)
    b = 0.0

    return w, b


# %% sigmaid function
def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z))

    return y_head


# %%

def forward_backward_propagation(w, b, x_train, y_train):
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train * np.log(y_head) - (1 - y_train) * np.log(1 - y_head)
    cost = (np.sum(loss)) / x_train.shape[1]

    derivative_weight = (np.dot(x_train, ((y_head - y_train).T))) / x_train.shape[1]
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    return cost, gradients


# %%

def update(w, b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []

    for i in range(number_of_iteration):
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)

        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]

        if i % 5 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after itaration %i: %f" % (i, cost))

    parameters = {"weight": w, "bias": b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()

    return parameters, gradients, cost_list


# %%

def predict(w, b, x_test):
    z = sigmoid(np.dot(w.T, x_test) + b)
    Y_prediction = np.zeros((1, x_test.shape[1]))

    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction


# %%

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    # initalize
    dimension = x_train.shape[0]  # that is 30
    w, b = initalize_weight_and_bias(dimension)
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)
    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)

    # print test errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))


# %%

logistic_regression(x_train, y_train, x_test, y_test, learning_rate=1, num_iterations=300)

k = 5;
