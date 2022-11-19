import os
import pandas as pd
files = os.listdir("data")
full_df = pd.DataFrame(columns=["Date"])
from datetime import datetime
first_date = []
def p2f(x):
    x = x.replace(',', '')
    return float(x.strip('%'))/100

for file in files:
    if ".csv" in file:
        df = pd.read_csv("data/"+file)
        coin = file.split()[0]
        try:
            df["Date"] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        except:pass
        try:
            df["Date"] = pd.to_datetime(df['Date'], format='%b %d, %Y')
        except: pass
        df = df[["Date","Open","High","Low","Vol.","Change %"]]
        df["returns_{}".format(coin)] = pd.Series([p2f(x) for x in df['Change %']], index = df.index)
        df = df.rename(columns={"Vol.":"Volume_{}".format(coin)})
        df = df.rename(columns={"Open":"Open_{}".format(coin)})
        df = df.rename(columns={"High":"High_{}".format(coin)})
        df = df.rename(columns={"Low":"Low_{}".format(coin)})
        first_avail_date = df['Date'].iloc[-1]
        tupl = tuple([coin, first_avail_date])
        first_date.append(tupl)
        if first_avail_date <= datetime.strptime('2018-07-01', '%Y-%m-%d'):
            full_df = full_df.merge(df, on="Date", how="outer")


full_df=full_df.dropna().reset_index(drop=True)
full_df.sort_values('Date', ascending=True)
print(full_df['Date'].iloc[-1])
print(full_df.head())
full_df.to_csv("price_data.csv")

