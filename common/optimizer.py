import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    #params is dict
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.learning_rate * grads[key]

#Momentum
class Momentum:
    def __init__(self, learning_rate=0.01, momentum= 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = None
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.learning_rate * grads[key]
            params[key] += self.v[key]


#AdaGrad
class AdaGrad:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.h = None


    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.learning_rate * grads[key]/(np.sqrt(self.h[key]+ 1e-7))

#RMSProp
class RMSProp:
    def __init__(self, learning_rate=0.01, decay=0.99):
        self.learning_rate = learning_rate
        self.decay = decay
        self.h = None


    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]*(1-self.decay) + self.decay * self.h[key]
            params[key] -= self.learning_rate * grads[key]/(np.sqrt(self.h[key]+ 1e-7))

#Adam
class Adam:
    def __init__(self, learning_rate=0.01, alpha1 = 0.9, alpha2 = 0.99):
        self.learning_rate = learning_rate
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.h = None
        self.v = None
        #iteration step
        self.t = 0
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            self.h = {}
            for key, val in params.items():
                self.v[key]  = np.zeros_like(val)
                self.h[key] = np.zeros_like(val)
        self.t += 1

        lr_t = self.learning_rate * np.sqrt(1 - self.alpha2 ** self.t) / (1 - self.alpha1 ** self.t)
        for key in params.keys():
            # self.v[key] = self.alpha1 * self.v[key] + (1 - self.alpha1) * grads[key]
            # self.h[key] = self.alpha2 * self.h[key] + (1 - self.alpha2) * grads[key]**2

            self.v[key]+= (1- self.alpha1)* (grads[key] - self.v[key])
            self.h[key]+= (1- self.alpha2)* (grads[key]**2 - self.h[key])

            params[key] -= lr_t * self.v[key]/(np.sqrt(self.h[key]+ 1e-7))


