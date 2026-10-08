import numpy as np
import matplotlib.pyplot as plt

from two_layer_net import TwoLayerNet
from common.load_data import get_data

x_train, t_train, x_test, t_test = get_data()

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

learning_rate = 0.01
batch_size = 100
num_epochs = 10

train_size = x_train.shape[0]
iter_per_epoch = max(train_size // batch_size, 1)   # 整除
iter_num = num_epochs * iter_per_epoch

train_loss_list = []
train_acc_list = []
test_acc_list = []

for i in range(iter_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.numerical_gradient(x_batch, t_batch)

    for key in grads:
        network.params[key] -= learning_rate * grads[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        epoch = i // iter_per_epoch
        print(f"epoch {epoch}, train acc: {train_acc:.4f}, test acc: {test_acc:.4f}")

x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()