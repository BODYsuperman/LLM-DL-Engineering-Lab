

import torch
import numpy as np
import matplotlib.pyplot as plt



def f(X):
    return 0.05*X[0]**2 + X[1]**2

def gradient_descent(X, optimizer, num_iters ):
    X_arr = X.detach().numpy().copy()
    for i in range (num_iters):
         y = f(X)
         y.backward()
         optimizer.step()
         optimizer.zero_grad()
         X_arr = np.vstack((X_arr, X.detach().numpy()))
    return X_arr


X = torch.tensor([-7, 2], requires_grad=True, dtype = torch.float32)

#define super params
lr = 1e-2
num_iters = 1000

#Optimizer compare
#1.SGD
X_clone = X.clone().detach().requires_grad_(True)
optimizer = torch.optim.SGD([X_clone], lr=lr)
X_arr1 = gradient_descent(X_clone, optimizer, num_iters)

plt.plot(X_arr1[:,0], X_arr1[:,1], 'r')
#2.Momentum

X_clone = X.clone().detach().requires_grad_(True)
optimizer = torch.optim.SGD([X_clone], lr=lr, momentum=0.9)
X_arr2 = gradient_descent(X_clone, optimizer, num_iters)

plt.plot(X_arr2[:,0], X_arr2[:,1], 'b')

x1_grid, x2_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))

y_grid = 0.05* x1_grid**2 + x2_grid**2
plt.contour(x1_grid, x2_grid, y_grid, levels=30, colors='k')
plt.legend(["SGD", "Momentum"])
plt.xlabel("x0")
plt.ylabel("x1")
plt.title("SGD vs Momentum")
plt.show()
