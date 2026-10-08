
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_data():

    data = pd.read_csv("../data/train.csv")
    X = data.drop("label", axis=1)
    y = data["label"]

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

    #feature engineering

    scaler = MinMaxScaler()

    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    train_y = train_y.values
    test_y = test_y.values

    return train_x, train_y, test_x, test_y

