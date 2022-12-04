from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

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
    train_X = df_coin.drop(columns="Log_RET_{}".format(coin))[0:nrows-1]
    train_y = df_coin["Log_RET_{}".format(coin)][1:nrows]

    #VALIDATION
    df_coin_test = df_test1[coin_cols]
    nrows = df_coin_test.shape[0]
    test_X = df_coin_test.drop(columns="Log_RET_{}".format(coin))[0:nrows-1]
    test_y = df_coin_test["Log_RET_{}".format(coin)][1:nrows]

    #XGBoost Model
    rf_model = RandomForestRegressor()
    test_params = {
        'max_depth':[6,8,10,12],
        'n_estimators': [50, 100, 500]
        }
    model = GridSearchCV(estimator = rf_model, param_grid = test_params)
    model.fit(train_X, train_y)
    depth = model.best_params_["max_depth"]
    n_estimators = model.best_params_["n_estimators"]
    model = RandomForestRegressor(max_depth=depth, n_estimators=n_estimators)
    model.fit(train_X, train_y)
    print(model.score(train_X, train_y))
    print(model.score(test_X, test_y))

    

