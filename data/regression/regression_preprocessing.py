import pandas as pd

stock = pd.read_csv('data/regression.csv')
print(stock.isna().sum())

