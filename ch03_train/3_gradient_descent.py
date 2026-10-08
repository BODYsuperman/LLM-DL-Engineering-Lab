import copy

import numpy as np
from matplotlib import pyplot as plt

from common.gradient import numerical_gradient


def gradient_descent(f, init_x, lr= 0.01, num_iter = 1000 ):

    x = init_x
    x_history = []

    for i in range(num_iter):
        #snapshot of x
        x_history.append(x.copy())

        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x, np.array(x_history)


def f(x):
    return x[0]**2 + x[1]**2
if __name__ == '__main__':
    #region
    # inner element is immutable
    a = [1, 2, 3]
    b = copy.copy(a)
    c = copy.deepcopy(a)


    b[0] = 99
    print(a)
    print(b)
    print(c)

    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    b = copy.copy(a)
    c= copy.deepcopy(a)

    b[0][0] = 99
    print(a)
    print(b)
    print(c)
    #endregion

    init_x = np.array([-3.0, 4.0])

    lr = 0.1
    num_iter = 20

    x, x_history = gradient_descent(f, init_x, lr, num_iter)

    print("the x is ", x)
    plt.plot([-5, 5], [0, 0], '--b')
    plt.plot([0, 0], [-5, 5], '--r')

    plt.scatter(x_history[:, 0], x_history[:, 1], c='r')

    plt.xlim([-3.5, 3.5])
    plt.ylim([-4.5, 4.5])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()











