from numpy import arange
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import numpy as np

data = pd.read_csv("features.csv")
open_cols = [col for col in data.columns if "Open" in col]
coins = [col.split("_")[1] for col in open_cols]

df_train = pd.read_csv("features_train.csv")
df_test1 = pd.read_csv("features_val.csv")
df_test2 = pd.read_csv("features_test.csv")

for coin in coins:
    coin_cols = [col for col in df_train.columns if coin in col]
    #TRAINING
    df_coin = df_train[coin_cols]
    nrows = df_coin.shape[0]
    train_X = df_coin.drop(columns="Log_RET_{}".format(coin))[65:nrows-1]
    train_y = df_coin["Log_RET_{}".format(coin)][66:nrows]

    #VALIDATION
    df_coin_test = df_test1[coin_cols]
    nrows = df_coin_test.shape[0]
    test_X = df_coin_test.drop(columns="Log_RET_{}".format(coin))[0:nrows-1]
    test_y = df_coin_test["Log_RET_{}".format(coin)][1:nrows]
    model = LinearRegression()
    model.fit(train_X, train_y)
    print(coin)
    y_pred = model.predict(test_X)
    y_pred = [np.nan] + list(y_pred)
    df_coin_test["predicted_y"] = y_pred
    df_coin_test.to_csv("{}_val_predictions.csv".format(coin))
    print(model.score(train_X, train_y))
    print(model.score(test_X, test_y))

    

