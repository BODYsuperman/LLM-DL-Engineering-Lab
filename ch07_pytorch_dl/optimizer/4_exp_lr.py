

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import SGD


def f(X):
    return 0.05*X[0]**2 + X[1]**2

X = torch.tensor([-7, 2], requires_grad=True, dtype = torch.float32)

lr = 0.9
num_iters = 500



optimizer = SGD([X], lr=lr)


lr_scheduler = ExponentialLR(optimizer,  gamma=0.99)

X_arr = X.detach().numpy().copy()
lr_list = []

for i in range(num_iters):
    y = f(X)

    y.backward()

    optimizer.step()

    optimizer.zero_grad()

    X_arr = np.vstack((X_arr, X.detach().numpy()))

    lr_list.append(optimizer.param_groups[0]['lr'])

    #update learning rate
    lr_scheduler.step()

#plot
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

x1_grid, x2_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
y_grid = 0.05* x1_grid**2 + x2_grid**2

#draw contour lines and X-point trajectory lines
ax[0].contour(x1_grid, x2_grid, y_grid, levels=30, colors='gray')
ax[0].plot(X_arr[:,0], X_arr[:,1], 'r')
ax[0].set_title('the process of gradient descent')

#the decay process of learning rate
ax[1].plot(lr_list, 'k')
ax[1].set_title('the decay process of learning rate')

plt.show()

