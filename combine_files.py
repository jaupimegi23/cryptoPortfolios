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
        df = df[["Date","Open"]]
        df = df.rename(columns={"Open":"price_{}".format(coin)})
        first_avail_date = df['Date'].iloc[-1]
        tupl = tuple([coin, first_avail_date])
        first_date.append(tupl)
        if first_avail_date <= datetime.strptime('2018-07-01', '%Y-%m-%d'):
            full_df = full_df.merge(df, on="Date", how="outer")


full_df=full_df.dropna().reset_index(drop=True)
print(full_df['Date'].iloc[-1])
print(full_df.head())
full_df.to_csv("price_data.csv")

