import torch
import torch.nn as nn
from torchsummary import summary


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 4)
        nn.init.xavier_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(4, 4)
        nn.init.kaiming_normal_(self.linear2.weight)
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.out(x)
        x = torch.softmax(x, dim=1)
        return x


def main():
    device = get_device()
    print(f"Using device: {device}")
    print(f"mps available: {torch.backends.mps.is_available()}")

    model = Model().to(device)

    x = torch.randn(10, 3, device=device)
    output = model(x)
    print("output:\n", output)

    print("\nnamed_parameters:")
    for name, param in model.named_parameters():
        print(name, tuple(param.shape))

    print("\nstate_dict keys:", list(model.state_dict().keys()))

    # torchsummary 只支持 cpu / cuda，这里强制用 cpu 看结构
    summary(model.to("cpu"), input_size=(3,), batch_size=10, device="cpu")


if __name__ == "__main__":
    main()