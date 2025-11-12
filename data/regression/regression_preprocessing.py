import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

stock = pd.read_csv('data/regression/regression.csv', parse_dates=['datetime'])
cols = [c for c in stock.columns if c != 'close'] + ['close']
stock = stock[cols]


print(stock.isna().sum(), "\n")

stock.sort_values('datetime', inplace=True)
stock.reset_index(drop=True, inplace=True)

dupes = stock.duplicated(subset=['datetime']).sum()
if dupes > 0:
    print(f"Usunięto {dupes} duplikatów po dacie.")
    stock = stock.drop_duplicates(subset=['datetime']).reset_index(drop=True)



print("Zakres dat:", stock['datetime'].min(), "→", stock['datetime'].max())
print("Wymiary zbioru:", stock.shape, "\n")

n = len(stock)

cut80 = int(n * 0.80)
train80 = stock.iloc[:cut80].copy()
test20  = stock.iloc[cut80:].copy()

cut70 = int(n * 0.70)
cut85 = int(n * 0.85)
train70      = stock.iloc[:cut70].copy()
test15       = stock.iloc[cut70:cut85].copy()
validation15 = stock.iloc[cut85:].copy()

target_col = 'close'
numeric_cols = [col for col in stock.select_dtypes(include=np.number).columns if col != target_col]

scaler_80 = StandardScaler()
train80_scaled = train80.copy()
test20_scaled  = test20.copy()

train80_scaled[numeric_cols] = scaler_80.fit_transform(train80[numeric_cols])
test20_scaled[numeric_cols]  = scaler_80.transform(test20[numeric_cols])

train80_scaled.to_csv('data/regression/train80.csv', index=False)
test20_scaled.to_csv('data/regression/test20.csv', index=False)

scaler_70 = StandardScaler()
train70_scaled      = train70.copy()
test15_scaled       = test15.copy()
validation15_scaled = validation15.copy()

train70_scaled[numeric_cols]      = scaler_70.fit_transform(train70[numeric_cols])
test15_scaled[numeric_cols]       = scaler_70.transform(test15[numeric_cols])
validation15_scaled[numeric_cols] = scaler_70.transform(validation15[numeric_cols])

train70_scaled.to_csv('data/regression/train70.csv', index=False)
test15_scaled.to_csv('data/regression/test15.csv', index=False)
validation15_scaled.to_csv('data/regression/validation15.csv', index=False)