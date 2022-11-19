import os
import pandas as pd
files = os.listdir("data")
full_df = pd.DataFrame(columns=["Date"])
import datetime
first_date = []
def p2f(x):
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
        df["returns_{}".format(coin)] = pd.Series([p2f(x) for x in df['Change %']], index = df.index)
        df = df[["Date","returns_{}".format(coin)]]
        full_df = full_df.merge(df, on="Date", how="outer")
        last_date = df['Date'].iloc[-1]
        tupl = tuple([coin, last_date])
        first_date.append(tupl)

#print(full_df.head())
full_df=full_df.dropna().reset_index(drop=True)
#print(full_df.head(-2))
print(sorted(first_date, key = lambda x: x[1]))
