import numpy as np

from common.functions import sigmoid, softmax, cross_entropy_error

#ReLU
class Relu():
    def __init__(self):
        #inner property, record all x such that x ≤ 0
        self.mask = None

    #forward
    def forward(self, x):
        #x = np.array([-1, 0, 2, -3])
        self.mask = (x <= 0)
        #mask = [True, True, False, True]
        out = x.copy()
        #out = [-1, 0, 2, -3]
        out[self.mask] = 0
        #out = [0, 0, 2, 0]
        return out

    def backward(self, dout):
        dx = dout.copy()  # 先复制
        dx[self.mask] = 0  # 只改副本
        return dx


class Sigmoid():
    def __init__(self):
        self.y = None
    def forward(self, x):
       y = sigmoid(x)
       self.y = y
       return y

    def backward(self, dout):
        dx = dout * (1.0 - self.y)*self.y
        return dx


class Affine:

    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None

        self.dW = None
        self.db = None
    def forward(self, x):

        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)

        self.x = x

        out = np.dot(x, self.W) + self.b

        return out
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)

        return dx


class SoftmaxWithLoss():
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        #if it's one-hot-vector encoding
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx /= batch_size
        return dx