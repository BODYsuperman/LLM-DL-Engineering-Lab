import numpy as np
from common.functions import sigmoid, softmax ,cross_entropy_error
from  common.gradient import numerical_gradient

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    #predict

    def forward(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = x@W1 + b1
        z1 = sigmoid(a1)
        a2 = z1@W2 + b2
        y = softmax(a2)

        return y

    #x:input t:supervised data:target
    def loss(self, x, t):
        y = self.forward(x)
        loss = cross_entropy_error(y, t)
        return loss

    def accuracy(self, x, t):
        y_prob = self.forward(x)
        y = np.argmax(y_prob, axis=1)
        accuracy = np.sum(y==t)/x.shape[0]
        return accuracy

    def numerical_gradient(self, x, t):
        loss_w = lambda _: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_w, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_w, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])
        return grads
    