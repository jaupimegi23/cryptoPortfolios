from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import os
import pandas as pd
data = pd.read_csv("features.csv")
open_cols = [col for col in data.columns if "Open" in col]
coins = [col.split("_")[1] for col in open_cols]

df_train = pd.read_csv("features_train.csv")
df_test1 = pd.read_csv("features_val.csv")
df_test2 = pd.read_csv("features_test.csv")

for coin in coins:
    coin_cols = [col for col in df_train.columns if coin in col]
    df_coin = df_train[coin_cols]
    nrows = df_coin.shape[0]
    train_X = df_coin.drop(columns="Log_RET_{}".format(coin))[0:nrows-1]
    train_y = df_coin["Log_RET_{}".format(coin)][1:nrows]

    df_coin_test = df_test1[coin_cols]
    test_X = df_coin_test.drop(columns="Log_RET_{}".format(coin))[0:nrows-1]
    test_y = df_coin_test["Log_RET_{}".format(coin)][1:nrows]

#[]
#test_X, test_y = df_test1[]
    xgb_model = xgb.XGBRegressor()
    test_params = {
        'max_depth':[2,4,8,12],
        "eta":[.35, .3, .25, .2, .15, .1, .05, .01, .005]
        }

    model = GridSearchCV(estimator = xgb_model,param_grid = test_params)
    model.fit(train_X, train_y)

    
    depth = model.best_params_["max_depth"]
    eta = model.best_params_["eta"]
    print(eta)
    
    model =xgb.XGBRegressor(max_depth=depth,eta=eta)
    model.fit(train_X, train_y)
    model.score(train_X, train_y)
    model.score()

