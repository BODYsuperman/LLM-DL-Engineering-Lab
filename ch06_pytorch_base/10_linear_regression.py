import torch

from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader


X = torch.randn(100, 1)
w = torch.tensor([5.5])
b = torch.tensor([15.0])
noise = torch.randn(100,1) * 0.1

y = w * X + b + noise

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

model = nn.Linear(in_features=1, out_features=1)

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 1e-3)

loss_list = []
epoch_num = 1000
for epoch in range(epoch_num):
    total_loss = 0
    #train_num = 0
    for x_train, y_train in dataloader:
        y_pre = model(x_train)
        loss_value = loss_fn(y_pre, y_train)
        total_loss += loss_value.item() * x_train.shape[0]
        #train_num += 1
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
    loss_list.append(total_loss / len(dataset))


print("训练后 w =", model.weight.item(), " b =", model.bias.item())

fig, ax = plt.subplots(1,2, figsize = (20,10))

ax[0].plot(loss_list)
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')

ax[1].scatter(X, y)

y_pred =model.weight.item() * X + model.bias.item()

ax[1].plot(X, y_pred, color='red')
plt.show()
