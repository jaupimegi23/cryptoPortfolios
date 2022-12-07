import pandas as pd
import os

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae

filename = ""

data = pd.read_csv("features.csv")
data = data.drop(data.columns[0], axis=1)
n_all = data.shape[0]
open_cols = [col for col in data.columns if "1Lag_OPEN_" in col]
coins = [col.split("_")[-1] for col in open_cols]
data = data.dropna()
print(coins)
for coin in coins:
    val_r2 = []
    test_r2 = []
    val_mse = []
    test_mse = []

    valid_preds_df = pd.read_csv("results/{}}.csv".format(filename))
    valid_preds = valid_preds_df["Log_RET_{}_pred".format(coin)].to_numpy()
    valid_actuals = valid_preds_df["Log_RET_{}".format(coin)].to_numpy()
    #NEED TO FIX THIS
    test_preds_df = pd.read_csv("results/{}}.csv".format(filename))
    test_preds = test_preds_df["Log_RET_{}_pred".format(coin)].to_numpy()
    test_actuals = test_preds_df["Log_RET_{}".format(coin)].to_numpy()

    val_r2 = r2_score(valid_actuals, valid_preds)
    val_mse = np.mean((valid_actuals-valid_preds)**2)
    val_mae = mae(valid_actuals,valid_preds)

    test_r2 = r2_score(valid_actuals, valid_preds)
    test_mse = np.mean((valid_actuals-valid_preds)**2)
    test_mae = mae(test_actuals,test_preds)
    
    print("Testing R2 for {} is {}".format(coin, test_r2))
    print("Testing MSE for {} is {}".format(coin, test_mse))
    print("Testing MAPE for {} is {}".format(coin, test_mae))