# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 20:10:56 2019

@author: mehmet.yilmaz
"""
# %% libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% read csv

data = pd.read_csv("data.csv")
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)  # isime yaramayan 2 adet column'u sildik
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]  # iki adet sınıfım vardı. Birincisi iyi huylu (M), digeri de kotu huylu (B).. Bunları 1 ve 0 olarak güncelledik.
print(data.info())

y = data.diagnosis.values;
x_data = data.drop(["diagnosis"], axis=1)

# %% normalization
# tüm veriler 0 ile hemen hemen 3000 birim arasındaydi. Bazı featuresler 0 ile
# 1 arasındakiken bazıları da 0 ile 3000 arasında değerler alıyordu
# Bu yüzden 3000 birim olan özellikle 0-1 arasındaki özelliğin etkisini ezebilirdi
# bu sebepten dolayı tüm featursler'ı 0 ile 1 arasında olacak şekilde normalize ettik
# normalizasyon = x - min(x) / max(x) - min(x) )
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

# %% train test split

# datam 529 adet sample den oluşuyor. Ancak bu datayı öğretsek de test edecek datam
# olması lazım. Bu yüzden eğitim verisi 529 adetin %80'i olacak, %20'lik kısmı da
# test data sı olacak
from sklearn.model_selection import train_test_split

# x -> featurs larım
# y -> class ım
# test_size -> x ve y 'nin yüzde 20 si test, yüzde 80 eğitim
# random_state -> random olarak böl ama 42 ID sini de unutma ve buna ata
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# transpozunu aldık sadece.
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train : ", x_train.shape)
print("x_test : ", x_test.shape)
print("y_train : ", y_train.shape)
print("y_test : ", y_test.shape)


# %% parameter initialize and sigmoid function

# dimension nedir? Bu 30 adet features 'ımız.. 30 tane ağırlık olması lazım.
def initialize_weights_and_bias(dimension):
    #   np.full((3,1),0.01)
    #   Out[53]:
    #    array([[0.01],
    #           [0.01],
    #           [0.01]])
    # 3'e 1'lik bir matris oluştur. Ve bu matrisin elemanları 0.01 den olussun
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return w, b


# w,b = initialize_weights_and_bias(30)

def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z))
    return y_head


# print(sigmoid(0))

# %%

def forward_backward_propagation(w, b, x_train, y_train):
    # forward propagation
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train * np.log(y_head) - (1 - y_train) * np.log(1 - y_head)
    cost = (np.sum(loss)) / x_train.shape[1]  # x_train.shape[1] = 455 ortalaması alındı

    # backward propagation
    derivative_weight = (np.dot(x_train, ((y_head - y_train).T))) / x_train.shape[1]
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    # gradients : weight ve bias 'ımızın türevlerini tutuyor içinde

    return cost, gradients


# %%

def update(w, b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []

    for i in range(number_of_iteration):
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)

        # update for weight
        w = w - learning_rate * gradients["derivative_weight"]
        # update for bians
        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 == 0:  # 10 adım da bir göster bana cost değerini. her adımda değil.
            cost_list2.append(cost)
            index.append(i)
            print("Cost after itaration %i: %f" % (i, cost))

    # ne kadar güncelleyeceğim peki? yani ne kadar iterate etmem lazım?
    # iste buna karar vermek icin cost function grafigini çizip bakarak...
    parameters = {"weight": w, "bias": b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()

    return parameters, gradients, cost_list


# %% prediction

def predict(w, b, x_test):
    z = sigmoid(np.dot(w.T, x_test) + b)
    Y_prediction = np.zeros((1, x_test.shape[1]))

    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction


# %% logistic regression
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    # initalize
    dimension = x_train.shape[0]  # that is 30
    w, b = initialize_weights_and_bias(dimension)
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)
    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)

    # print test errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))


logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 3, num_iterations = 30)

# %% sklearn with LR
# yukarida yaptigimiz tum adimlarin aynisini asagidaki 4 kod yapmaktadir... :)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T, y_train.T)
print("test accuracy: {} %".format(lr.score(x_test.T, y_test.T)))