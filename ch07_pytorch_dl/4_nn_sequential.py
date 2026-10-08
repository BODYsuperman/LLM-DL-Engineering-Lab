

import torch
import torch.nn as nn
from torchsummary import summary

x = torch.randn(10, 3)

model = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.Softmax(dim=1),

)


def init_params(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0.1)
model.apply(init_params)

output = model(x)
print(output)

summary(model, input_size=(3,), batch_size=10)

