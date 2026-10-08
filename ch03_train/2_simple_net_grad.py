import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def forward(self, X):
        a = X @ self.W
        y = softmax(a)
        return y

    #x is input , y is target
    def loss(self, x, t):
        y = self.forward(x)
        loss = cross_entropy_error(y, t)
        return loss

if __name__ == '__main__':
    x = np.array([0.6, 0.9])
    y = np.array([0, 0, 1])
    net = SimpleNet()

    f= lambda _ : net.loss(x, y)
    dW = numerical_gradient(f, net.W)
    print(dW)