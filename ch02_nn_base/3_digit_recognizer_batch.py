

import numpy as np
import pandas as pd

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from common.functions import sigmoid, softmax


#read data
def get_data():

    data = pd.read_csv("../data/train.csv")
    X = data.drop("label", axis=1)
    y = data["label"]

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

    #feature engineering

    scaler = MinMaxScaler()

    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    return test_x, test_y


def init_network():
    network = joblib.load("../data/nn_sample")
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

X, y = get_data()

network = init_network()

print(network['W1'].shape)
print(network['W2'].shape)
print(network['W3'].shape)
print(network['b1'].shape)
print(network['b2'].shape)
print(network['b3'].shape)


batch_size = 100
acc_cnt = 0
n = X.shape[0]


for i in range(0, n, batch_size):
    x_batch = X[i:i + batch_size]
    y_batch = predict(network, x_batch)

    y_pred = np.argmax(y_batch, axis=1)
    acc_cnt += np.sum(y_pred == y[i:i + batch_size])

print("Accuracy: ", acc_cnt / n)


