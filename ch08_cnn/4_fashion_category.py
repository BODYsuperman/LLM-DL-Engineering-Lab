

import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


fashion_mnist_train = pd.read_csv('../data/fashion-mnist_train.csv')
fashion_mnist_test = pd.read_csv('../data/fashion-mnist_test.csv')

#convert origin data(N*1*784) to tensor (N*1*28*28)
"""
gray pic：(N, 1, 28, 28)
RGB ：(N, 3, H, W)
"""
X_train = torch.tensor(fashion_mnist_train.iloc[:, 1:].values, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_train = torch.tensor(fashion_mnist_train.iloc[:,0].values, dtype=torch.int64)

X_test = torch.tensor(fashion_mnist_test.iloc[:, 1:].values, dtype=torch.float32).reshape(-1, 1, 28, 28)
#must define as torch.int64 that means torch.long like nn.CrossEntropyLoss
y_test = torch.tensor(fashion_mnist_test.iloc[:,0].values, dtype=torch.int64)

# plt.imshow(X_train[12345, 0,:,:], cmap='gray')
# plt.show()

print(y_train[12345])

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

model = nn.Sequential(
    nn.Conv2d(1, 6, 5, stride=1, padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, 5,  stride=1, padding=0),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),

    nn.Flatten(),

    nn.Linear(400, 120),
    nn.Sigmoid(),

    nn.Linear(120, 84),
    nn.Sigmoid(),

    nn.Linear(84, 10),
)
x = torch.rand(size = (1, 1, 28, 28), dtype = torch.float)
for i, layer in enumerate(model):
    x = layer(x)
    print(f"Layer {i:2d} | {layer.__class__.__name__:12s} | 输出 shape: {list(x.shape)}")


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
#hyperparameter
lr = 1e-3
epoch_num = 20
batch_size = 64

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def train_test(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device ):
    def init_params(layer):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.1)

    model.apply(init_params)
    model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch_num):


        train_loss_accumulate = 0
        train_correct_num = 0
        for batch_idx ,(X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            output = model(X).squeeze(1)
            loss_value = loss(output, y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            train_loss_accumulate += loss_value.item()


            pred = output.argmax(dim=1)
            train_correct_num += pred.eq(y).sum()

            print(f"\rEpoch:{epoch + 1:0>2}[{'='*int((batch_idx+1)/len(train_loader)*50)}]", end="")

        this_train_loss = train_loss_accumulate / len(train_dataset)
        this_train_acc = train_correct_num / len(train_dataset)

        model.eval()

        test_correct_num = 0

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                pred = output.argmax(dim=1)
                test_correct_num += pred.eq(y).sum()
            this_test_acc = test_correct_num / len(test_dataset)

            print(f"train_loss:{this_train_loss:.4f}, train_acc:{this_train_acc:.4f}, test_acc:{this_test_acc:.4f}")


train_test(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device)

plt.imshow(X_test[666, 0, :, :], cmap='gray')
plt.show()

print(y_test[666])

output = model(X_test[666].unsqueeze(0).to(device))
y_pred = output.argmax(dim=1)

print(y_pred)