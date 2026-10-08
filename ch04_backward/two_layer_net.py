import numpy as np
from common.functions import sigmoid, softmax ,cross_entropy_error
from common.gradient import numerical_gradient
from common.layers import  *
from collections import OrderedDict

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)


        #define layer structure
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()


    #predict

    def forward(self, x):

        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    #x:input t:supervised data:target
    def loss(self, x, t):
        y = self.forward(x)
        loss = self.lastLayer.forward(y, t)
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
    def gradient(self, x, t):
        #forward full process until loss
        self.loss(x, t)
        #backward
        dy = 1
        dy = self.lastLayer.backward(dy)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dy = layer.backward(dy)
        grads = {}

        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads

    