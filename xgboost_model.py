from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

data = pd.read_csv("features.csv")
data = data.drop(data.columns[0], axis=1)
n_all = data.shape[0]
open_cols = [col for col in data.columns if "Open" in col]
coins = [col.split("_")[1] for col in open_cols]
print(coins)
data = data[65:].reset_index(drop=True)

all_data = "TRUE"
lookback = 100
lookforward = 1

val_size = lookforward * 10
test_size = lookforward * 10

train = data[:-1][:-val_size-test_size]
n_train = len(train)

val = data[:-1][-val_size-test_size:-test_size]
n_val = len(val)

test = data[:-1][-test_size:]
n_test = len(test)

for coin in coins:
    i = 0
    coin_cols = [col for col in data.columns if coin in col]
    df_coin = data[coin_cols]
    x = df_coin.drop(columns="Log_RET_{}".format(coin))[:-1]
    y = df_coin["Log_RET_{}".format(coin)][1:]

    nume = []
    deno = []
    y_valid_hat_coin = np.array([])
    # Validation
    if all_data == "TRUE":
        for i in range(0, n_val, lookforward):
            x_train = x.iloc[0:n_train + i]
            y_train = y.iloc[0:n_train + i]

            #RUN CV ONLY IF FIRST TIME DOING THE REGRESSION
            if i == 0:
                xgb_model = XGBRegressor()
                test_params = {
                    'max_depth':[6,8,10,12],
                    }
                model = GridSearchCV(estimator = xgb_model,param_grid = test_params)
                model.fit(x_train, y_train)
                depth = model.best_params_["max_depth"]
                print(depth)
                model = XGBRegressor(max_depth=depth,eta=.05)
           
            x_valid = x.iloc[n_train + i :n_train + i + lookforward]
            y_valid = y.iloc[n_train + i :n_train + i + lookforward]
            model.fit(x_train, y_train)
            y_valid_hat_wind = model.predict(x_valid)
            y_valid_hat_coin = np.append(y_valid_hat_coin, y_valid_hat_wind, axis=0)
            nume.append((y_valid_hat_wind-y_valid.values)**2)
            deno.append((y_valid.values-y_train.mean())**2)

    val["Log_RET_{}_pred".format(coin)] = y_valid_hat_coin

    r2 = r2_score(val["Log_RET_{}".format(coin)][:-1], val["Log_RET_{}_pred".format(coin)][1:])
    print("Validation R2 for {} is {}".format(coin, r2))

    nume = np.array(nume)
    nume = nume.reshape(nume.shape[0]*nume.shape[1],)
    deno = np.array(deno)
    deno = deno.reshape(deno.shape[0]*deno.shape[1],)
    osr2 = 1-sum(nume)/sum(deno)
    print("Validation OSR2 for {} is {}".format(coin, osr2))

    mse = np.mean((val["Log_RET_{}".format(coin)][:-1] - val["Log_RET_{}_pred".format(coin)][1:])**2)
    print("Validation MSE for {} is {}".format(coin, mse))

    # Test
    y_test_hat_coin = np.array([])
    if all_data == "TRUE":
        for i in range(0, n_test, lookforward):
            x_train = x.iloc[0:n_train + n_val + i]
            y_train = y.iloc[0:n_train + n_val + i]
            x_test = x.iloc[n_train + n_val + i :n_train + n_val + i + lookforward]
            y_test = y.iloc[n_train + n_val + i :n_train + n_val + i + lookforward]
            model.fit(x_train, y_train)
            y_test_hat_wind = model.predict(x_test)
            y_test_hat_coin = np.append(y_test_hat_coin, y_test_hat_wind, axis=0)

    test["Log_RET_{}_pred".format(coin)] = y_test_hat_coin


# Save predictions
val.to_csv("results/sparse_val.csv")
test.to_csv("results/sparse_test.csv")



