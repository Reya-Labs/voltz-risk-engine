import pandas as pd

from utils import date_to_unix_time

df = pd.read_csv("rni_historical_data/aave_usdc.csv")

print(df)

df_unix_time = date_to_unix_time(df, df)

print(df_unix_time)