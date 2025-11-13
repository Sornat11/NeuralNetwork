import pandas as pd
from correlation import smart_correlation_matrix
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

loan_data = pd.read_csv("data/classification_our/loan_approval.csv")
print(loan_data.head())

print(loan_data.isna().sum() / len(loan_data))

# brak braków.
print(loan_data.columns)

# id nie ma znaczenia predykcyjnego, więc usuwamy
loan_data.drop('customer_id', axis=1, inplace=True)

# mamy 3 zmienne objaśniające kategoryczne: occupation_status, loan_intent, product_type, pozostałe są liczbowe

categorical_cols = ['occupation_status', 'loan_intent', 'product_type']
for col in categorical_cols:
    print(f"\nKolumna: {col}")
    print(loan_data[col].unique())

# zmienne kategoryczne mają niewiele unikalnych wartości, więc można je zakodować one-hot encodingiem

# smart_correlation_matrix(loan_data, figsize=(10,8))

# usuwamy zmienne, które mają niską korelację z targetem a także te silnie powiązane z wiekiem
loan_data.drop(['occupation_status', 'loan_amount','savings_assets','current_debt','product_type','loan_intent','interest_rate', 'years_employed','credit_history_years'], axis=1, inplace=True)

# smart_correlation_matrix(loan_data, figsize=(10,8))

print(loan_data.head())



# finalnie mamy 9 zmiennych objaśniających numerycznych oraz target 'loan_status'
print(loan_data.columns)
num_cols = [
    'age', 'annual_income', 'credit_score', 'defaults_on_file',
    'delinquencies_last_2yrs', 'derogatory_marks', 'debt_to_income_ratio',
    'loan_to_income_ratio', 'payment_to_income_ratio'
]

print('Ile klasy 1:')
print(loan_data['loan_status'].sum() / len(loan_data))

for col in num_cols:
    q1, q3 = loan_data[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (loan_data[col] < lower) | (loan_data[col] > upper)

    outlier_count = mask.sum()
    outlier_pct = (outlier_count / len(loan_data)) * 100
    outliers_class1 = loan_data.loc[mask, 'loan_status'].sum()
    class1_pct = (outliers_class1 / outlier_count * 100) if outlier_count > 0 else 0

    print(f"{col:30} → {outlier_count:5d} outlierów ({outlier_pct:5.2f}%) | klasa 1: {outliers_class1:4d} ({class1_pct:5.2f}%)")

# Dla większości klas outliery stanowią mały procent, więc nie będą stanowić dużego problemu.
# wyjątki to annual_income, defaults_on_file, deliquencies_last_2yrs, derogatory_marks
# spośród tych 4 klas żadna nie ma jednak rozkładu zmiennej objaśnianej typowego dla całego zbioru. Outliery te mogą więc być istotne. W celu redukcji wpływu wykorzystamy logarytmowanie tam, gdzie to możliwe.

cols_to_log = ['annual_income', 'defaults_on_file', 'derogatory_marks', 'delinquencies_last_2yrs']
loan_data[cols_to_log] = loan_data[cols_to_log].apply(lambda x: np.log1p(x))

# Undersampling klasy większościowej (1)
class0 = loan_data[loan_data['loan_status'] == 0]
class1 = loan_data[loan_data['loan_status'] == 1]

# Ile jest klasy mniejszościowej (0)
n_minority = len(class0)

# Undersampling klasy 1
class1_downsampled = resample(
    class1,
    replace=False,
    n_samples=n_minority,
    random_state=42
)

# Połączenie i przemieszanie
loan_balanced = pd.concat([class0, class1_downsampled], axis=0).sample(frac=1, random_state=42)

target_col = 'loan_status'

# cechy numeryczne (bez targetu)
numeric_cols = [c for c in loan_balanced.select_dtypes(include=np.number).columns if c != target_col]

# X / y
X = loan_balanced.drop(columns=[target_col])
y = loan_balanced[target_col]

# ===== WARIANT 1: 80 / 20 =====
X_train80, X_test20, y_train80, y_test20 = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler_80 = StandardScaler()
X_train80_s = X_train80.copy()
X_test20_s  = X_test20.copy()

X_train80_s[numeric_cols] = scaler_80.fit_transform(X_train80[numeric_cols])
X_test20_s[numeric_cols]  = scaler_80.transform(X_test20[numeric_cols])

train80 = pd.concat([X_train80_s.reset_index(drop=True), y_train80.reset_index(drop=True)], axis=1)
test20  = pd.concat([X_test20_s.reset_index(drop=True),  y_test20.reset_index(drop=True)], axis=1)

# target na końcu
order_cols = [c for c in train80.columns if c != target_col] + [target_col]
train80 = train80[order_cols]
test20  = test20[order_cols]

train80.to_csv('data/classification_our/train80.csv', index=False)
test20.to_csv('data/classification_our/test20.csv', index=False)

# ===== WARIANT 2: 70 / 15 / 15 =====
X_train70, X_temp, y_train70, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_test15, X_val15, y_test15, y_val15 = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

scaler_70 = StandardScaler()
X_train70_s = X_train70.copy()
X_test15_s  = X_test15.copy()
X_val15_s   = X_val15.copy()

X_train70_s[numeric_cols] = scaler_70.fit_transform(X_train70[numeric_cols])
X_test15_s[numeric_cols]  = scaler_70.transform(X_test15[numeric_cols])
X_val15_s[numeric_cols]   = scaler_70.transform(X_val15[numeric_cols])

train70      = pd.concat([X_train70_s.reset_index(drop=True), y_train70.reset_index(drop=True)], axis=1)
test15       = pd.concat([X_test15_s.reset_index(drop=True),  y_test15.reset_index(drop=True)], axis=1)
validation15 = pd.concat([X_val15_s.reset_index(drop=True),   y_val15.reset_index(drop=True)], axis=1)

train70      = train70[order_cols]
test15       = test15[order_cols]
validation15 = validation15[order_cols]

train70.to_csv('data/classification_our/train70.csv', index=False)
test15.to_csv('data/classification_our/test15.csv', index=False)
validation15.to_csv('data/classification_our/validation15.csv', index=False)