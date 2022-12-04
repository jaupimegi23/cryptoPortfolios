from numpy import arange
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import numpy as np

data = pd.read_csv("features.csv")
open_cols = [col for col in data.columns if "Open" in col]
coins = [col.split("_")[1] for col in open_cols]
print(coins)
df_train = pd.read_csv("features_train.csv")
df_test1 = pd.read_csv("features_val.csv")
df_test2 = pd.read_csv("features_test.csv")
ALL_COIN_PREDS = []
for coin in coins:
    coin_cols = [col for col in df_train.columns if coin in col]
    #TRAINING
    df_coin = df_train[coin_cols]
    nrows = df_coin.shape[0]
    train_X = df_coin.drop(columns="Log_RET_{}".format(coin))[65:nrows-1]
    train_y = df_coin["Log_RET_{}".format(coin)][66:nrows]
    df_coin_test = df_test1[coin_cols]

    nrows = df_coin_test.shape[0]
    test_X_ALL = df_coin_test.drop(columns="Log_RET_{}".format(coin))[0:nrows-1]
    test_y_ALL = df_coin_test["Log_RET_{}".format(coin)][1:nrows]
    model = LinearRegression()
    model.fit(train_X, train_y)
    print(coin)
    print(model.score(train_X, train_y))
    nume = []
    deno = []
    for i in range(df_coin_test.shape[0]-1):
        if i != 0:
            train_X = pd.concat([train_X,test_X_ALL.iloc[[i-1]]])
            train_y = pd.concat([train_y,test_y_ALL.iloc[[i-1]]])
            model.fit(train_X, train_y)
        train_y_mean = train_y.mean()
        test_X = test_X_ALL[i:i+1]
        test_y = test_y_ALL[i:i+1]
        y_pred = model.predict(test_X)
        nume.append((y_pred-train_y_mean)[0]**2)
        deno.append((test_y.mean()-train_y_mean)**2)
    #OSR2
    print(1-sum(nume)/sum(deno))

    

