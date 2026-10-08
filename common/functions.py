


#1. Binary Step function

def step_function(s_x):
    return 1 if s_x > 0 else 0


import  numpy as np

def step_function2(f_x):
    return np.array(f_x > 0, dtype=int)


#2. Sigmoid function

def sigmoid(smx):
    return 1 / (1 + np.exp(-smx))
def sigmoid_derivative(sdx):
    return sigmoid(sdx) * (1 - sigmoid(sdx))


#3. ReLu function

def relu(smx):
    return np.maximum(0, smx)

def relu_derivative(sdx):
    return relu(sdx) * (1 - relu(sdx))


def softmax(smx):

    if smx.ndim == 2:
        smx = smx - np.max(smx, axis=1, keepdims=True)

        exp_smx = np.exp(smx)
        return exp_smx / np.sum(exp_smx, axis=1, keepdims=True)

    smx = smx - np.max(smx)
    exp_smx = np.exp(smx)
    return exp_smx / np.sum(exp_smx)


def identity(x):
    return x
def identity_derivative(x):
    return 1


#MSE
def mean_squared_error(y, t):
    return np.sum((y - t) ** 2)

#cross_entropy_error
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    if t.size == y.size:
        t = t.argmax(axis=1)
    n = y.shape[0]

    return -np.sum(np.log(y[np.arange(n), t])) / n

if __name__ == '__main__':
    x = np.array([0, 1, 2, 3, 4, 5, 6, -3,-7, -8, -9, -10])
    print(x.shape)

    print(step_function2(x))

    print(sigmoid(x))

    print("*"*80)
    print(sigmoid_derivative(x))

    print("*"*80)

    print(np.tanh(x))
    print("*"*80)

    print(relu(x))
