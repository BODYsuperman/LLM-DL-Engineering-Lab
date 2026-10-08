import numpy as np


#numerical differential

def numerical_diff(f, x):

    h = 1e-4
    #central differential
    return (f(x + h) - f(x-h)) / (h*2)



#x is a vector
def _numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.size):
        tmp_x = x[i]
        x[i] = tmp_x + h

        fxh1 = f(x) #f(x + h)

        x[i] = tmp_x - h

        fxh2 = f(x)  # f(x - h)

        grad[i] = (fxh1 - fxh2)/(2*h)

        x[i] = tmp_x

    return grad


#X is a matrix
def numerical_gradient(f, X):

    if(X.ndim == 1):
        return _numerical_gradient(f, X)
    else:
        grad = np.zeros_like(X)
        for i, x in enumerate(X):
            grad[i] = _numerical_gradient(f, x)
        return grad




