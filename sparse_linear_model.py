from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE

data = pd.read_csv("features.csv")
open_cols = [col for col in data.columns if "Open" in col]
coins = [col.split("_")[1] for col in open_cols]


df_train = pd.read_csv("features_train.csv")
df_test1 = pd.read_csv("features_val.csv")
df_test2 = pd.read_csv("features_test.csv")


for coin in coins:
    print(coin)
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
    
    lm = LinearRegression()
    lm.fit(train_X, train_y)
    rfe = RFE(lm)             

    hyper_params = [{'n_features_to_select': list(range(5, 21))}]
    folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

    model_cv = GridSearchCV(estimator = rfe, 
                        param_grid = hyper_params, 
                        scoring= 'r2', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

    model_cv.fit(train_X, train_y)
    print(model_cv.best_params_["n_features_to_select"])

    print(model_cv.score(train_X, train_y))
    print(model_cv.score(test_X, test_y))

    

