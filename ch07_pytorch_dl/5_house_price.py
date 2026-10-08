
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader



def create_dataset():
    """Create dataset"""

    data = pd.read_csv('../data/house_prices.csv')
    data.drop(["Id"], axis=1, inplace=True)

    X= data.drop("SalePrice", axis=1)
    y = data["SalePrice"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #feature_engineering(transform)
    ## classify data as numerical and category
    numerical_features = X.select_dtypes(exclude=["object"]).columns
    categorical_features = X.select_dtypes(include=["object", "str"]).columns
    ##define column transform
    numerical_transformer = Pipeline(steps=[
        ("fillna", SimpleImputer(strategy="mean")),
        ("std", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("fillna", SimpleImputer(strategy="constant", fill_value="NaN")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(

        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)

    x_train = pd.DataFrame(x_train.toarray(), columns=preprocessor.get_feature_names_out())
    x_test = pd.DataFrame(x_test.toarray(), columns=preprocessor.get_feature_names_out())

    #construct tensor dataset
    train_dataset = TensorDataset(torch.tensor(x_train.values).float(), torch.tensor(y_train.values).float())

    test_dataset = TensorDataset(torch.tensor(x_test.values).float(), torch.tensor(y_test.values).float())

    return train_dataset, test_dataset, x_train.shape[1]

train_dataset, test_dataset, feature_num = create_dataset()
#create model

model = nn.Sequential(
    nn.Linear(in_features=feature_num, out_features=128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(in_features=128, out_features=1),
)

def log_rmse(pred, target):

    pred.squeeze_()
    pred = torch.clamp(pred, 1, float("inf"))
    mse = nn.MSELoss()
    return torch.sqrt(mse(torch.log(pred) , torch.log(target)))

def train_test(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device ):
    def init_params(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.1)

    model.apply(init_params)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_list = []
    test_loss_list = []

    for epoch in range(epoch_num):
        model.train()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        train_loss_accumulate = 0
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)

            output = model(X).squeeze(1)
            loss = log_rmse(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_accumulate += loss.item()

        this_train_loss = train_loss_accumulate / len(train_loader)
        train_loss_list.append(this_train_loss)

        model.eval()
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
        test_loss_accumulate = 0

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X).squeeze(1)
                loss = log_rmse(y_pred, y)
                test_loss_accumulate += loss.item()
            this_test_loss = test_loss_accumulate / len(test_loader)
            test_loss_list.append(this_test_loss)

        print(f"epoch:{epoch+1 }, train_loss:{this_train_loss:.4f}, test_loss:{this_test_loss:.4f}")

    return train_loss_list, test_loss_list
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
#hyperparameter
lr = 1e-3
epoch_num = 2000
batch_size = 64

train_loss_list , test_loss_list = train_test(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device)

plt.plot(train_loss_list,'r-', label="train loss", linewidth=3)
plt.plot(test_loss_list,'k--', label="test loss", linewidth=2)

plt.legend()
plt.show()