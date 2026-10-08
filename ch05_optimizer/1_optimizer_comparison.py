import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *


def f(x, y):
    return x**2/20 + y**2

def f_grad(x, y):
    return x/10, 2*y


init_pos = (-7.0, 2.0)
params = {}
grads = {}

optimizers = OrderedDict()
optimizers['SGD'] = SGD(learning_rate=0.9)
optimizers['Momentum'] = Momentum(learning_rate=0.11, momentum=0.85)
optimizers['AdaGrad'] = AdaGrad(learning_rate=1.5)
optimizers['Adam'] = Adam(learning_rate=0.5, alpha1=0.5)

idx = 1

for key in optimizers:
    opt = optimizers[key]          # 用不同的变量名
    x_history = []
    y_history = []

    params['x'], params['y'] = init_pos[0], init_pos[1]

    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])

        grads['x'], grads['y'] = f_grad(params['x'], params['y'])
        opt.update(params, grads)

    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    x, y = np.meshgrid(x, y)
    z = f(x, y)
    z[z > 7] = 0

    plt.subplot(2, 2, idx)
    idx += 1

    plt.contour(x, y, z, levels=20)
    plt.plot(0, 0, '+')
    plt.plot(x_history, y_history, 'o-', color='red',
             markersize=2, label=key)
    plt.xlim(-10, 10)
    plt.ylim(-5, 5)
    plt.legend(loc="best")

plt.show()