import pandas as pd
import numpy as np
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from correlation import smart_correlation_matrix

# === 0) Wczytanie i porządek w nazwach kolumn ===
adult_data = pd.read_csv('data/classification/classification.csv')

# mapowanie nazw -> snake_case bez kropek
rename_map = {
    'education.num': 'education_num',
    'marital.status': 'marital_status',
    'capital.gain': 'capital_gain',
    'capital.loss': 'capital_loss',
    'hours.per.week': 'hours_per_week',
    'native.country': 'native_country',
    # pozostałe już są OK: age, workclass, fnlwgt, education, occupation, relationship, race, sex, income
}
adult_data.rename(columns=rename_map, inplace=True)

# Zamiana '?' na NA
adult_data.replace('?', pd.NA, inplace=True)

# Podgląd braków (proporcje)
print(adult_data.isna().sum() / len(adult_data))
print(adult_data.columns)

# Są braki - w kolumnach workclass, occupation, native_country obczaimy czy można wywalić te wiersze.
cols_with_nans = ['workclass', 'occupation', 'native_country']

print("Cały zbiór:")
print(adult_data['income'].value_counts(normalize=True))

for col in cols_with_nans:
    mask = adult_data[col].isna()
    print(f"\nKolumna: {col}")
    print("\nWiersze z brakami:")
    print(adult_data.loc[mask, 'income'].value_counts(normalize=True))

# native_country ma podobny rozkład klasy docelowej, więc można usunąć wiersze z brakami
# workclass i occupation mają inny rozkład, więc usunięcie wierszy z brakami może zaburzyć rozkład klasy docelowej

# 1) Dropujemy kategoryczne 'education' (duplikat information z education_num)
adult_data.drop('education', axis=1, inplace=True)

# 2) Usuwamy wiersze z brakami tylko w native_country
adult_data.dropna(subset=['native_country'], inplace=True)

# Podgląd unikalnych wartości w kategoriach (po rename)
for col in adult_data.select_dtypes(include='object').columns:
    print(f"\nKolumna: {col}")
    print(adult_data[col].unique())

# Uzupełnienie braków
adult_data['occupation'] = adult_data['occupation'].fillna('Unknown')
adult_data['workclass'] = adult_data['workclass'].fillna('Unknown')

# Proste mapowania binarne
adult_data['sex'] = adult_data['sex'].map({'Male': 1, 'Female': 0})
adult_data['income'] = adult_data['income'].apply(lambda x: 1 if x == '>50K' else 0)
adult_data['native_country'] = adult_data['native_country'].apply(lambda x: 1 if x == 'United-States' else 0)

print(adult_data.head())

# Rozkład workclass przed grupowaniem
print(adult_data['workclass'].value_counts(normalize=True) * 100)

corr_all = smart_correlation_matrix(adult_data)
# z automatu drop tych, które mają <0.15 korelacji z income
adult_data.drop(['native_country', 'fnlwgt','race'], axis=1, inplace=True)
corr_all = smart_correlation_matrix(adult_data)


# Grupowanie workclass → 5 klas
workclass_map = {
    'Private': 'Private',
    'Federal-gov': 'Government',
    'Local-gov': 'Government',
    'State-gov': 'Government',
    'Self-emp-not-inc': 'Self-employed',
    'Self-emp-inc': 'Self-employed',
    'Without-pay': 'Not-working',
    'Never-worked': 'Not-working',
    'Unknown': 'Unknown'
}
adult_data['workclass'] = adult_data['workclass'].map(workclass_map)
print(adult_data['workclass'].value_counts(normalize=True) * 100)

# Rozkład marital_status przed grupowaniem
print(adult_data['marital_status'].value_counts(normalize=True) * 100)

# Grupowanie marital_status → Married / Never-married / Separated
marital_map = {
    'Married-civ-spouse': 'Married',
    'Never-married': 'Never-married',
    'Divorced': 'Separated',
    'Separated': 'Separated',
    'Widowed': 'Separated',
    'Married-spouse-absent': 'Separated',
    'Married-AF-spouse': 'Separated'
}
adult_data['marital_status'] = adult_data['marital_status'].map(marital_map)
print(adult_data['marital_status'].value_counts(normalize=True) * 100)

# Target encoding dla occupation (szybka wersja)
print(adult_data['occupation'].value_counts(normalize=True) * 100)
global_mean = adult_data['income'].mean()
occ_means = adult_data.groupby('occupation')['income'].mean()
adult_data['occupation'] = adult_data['occupation'].map(occ_means).fillna(global_mean)

# One-hot encoding dla wybranych kolumn (JUŻ po rename)
cols_to_encode = ['marital_status', 'relationship', 'workclass']

adult_data = pd.get_dummies(
    adult_data,
    columns=cols_to_encode,
    prefix=cols_to_encode,
    dtype=int,
    drop_first=False
)

print(adult_data.columns)
print(adult_data.head())
adult_data['capital_net_log'] = np.sign(adult_data['capital_gain'] - adult_data['capital_loss']) * np.log1p(np.abs(adult_data['capital_gain'] - adult_data['capital_loss']))
adult_data.drop(['capital_gain', 'capital_loss'], axis=1, inplace=True)
print(adult_data['income'].value_counts(normalize=True) * 100)

majority = adult_data[adult_data['income'] == 0]
minority = adult_data[adult_data['income'] == 1]

majority_downsampled = resample(
    majority,
    replace=False,
    n_samples=len(minority),  
    random_state=42
)

adult_balanced = pd.concat([majority_downsampled, minority]).sample(frac=1, random_state=42).reset_index(drop=True)
print(adult_balanced['income'].value_counts(normalize=True))
print(adult_balanced.shape)
print(adult_balanced.columns)

to_be_standardized = ['age', 'education_num', 'hours_per_week', 'capital_net_log']

X = adult_balanced.drop(columns=['income'])
y = adult_balanced['income']
X_train80, X_test20, y_train80, y_test20 = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# --- Standaryzacja tylko wybranych kolumn ---
scaler_80 = StandardScaler()
X_train80_scaled = X_train80.copy()
X_test20_scaled = X_test20.copy()

X_train80_scaled[to_be_standardized] = scaler_80.fit_transform(X_train80[to_be_standardized])
X_test20_scaled[to_be_standardized]  = scaler_80.transform(X_test20[to_be_standardized])

train80 = pd.concat([X_train80_scaled.reset_index(drop=True),
                     y_train80.reset_index(drop=True)], axis=1)
test20  = pd.concat([X_test20_scaled.reset_index(drop=True),
                     y_test20.reset_index(drop=True)], axis=1)

train80.to_csv('data/classification/train80.csv', index=False)
test20.to_csv('data/classification/test20.csv', index=False)


X_train70, X_temp, y_train70, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_test15, X_val15, y_test15, y_val15 = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# --- Standaryzacja tylko wybranych kolumn ---
scaler_70 = StandardScaler()
X_train70_scaled = X_train70.copy()
X_test15_scaled  = X_test15.copy()
X_val15_scaled   = X_val15.copy()

X_train70_scaled[to_be_standardized] = scaler_70.fit_transform(X_train70[to_be_standardized])
X_test15_scaled[to_be_standardized]  = scaler_70.transform(X_test15[to_be_standardized])
X_val15_scaled[to_be_standardized]   = scaler_70.transform(X_val15[to_be_standardized])

train70      = pd.concat([X_train70_scaled.reset_index(drop=True),
                          y_train70.reset_index(drop=True)], axis=1)
test15       = pd.concat([X_test15_scaled.reset_index(drop=True),
                          y_test15.reset_index(drop=True)], axis=1)
validation15 = pd.concat([X_val15_scaled.reset_index(drop=True),
                          y_val15.reset_index(drop=True)], axis=1)

train70.to_csv('data/classification/train70.csv', index=False)
test15.to_csv('data/classification/test15.csv', index=False)
validation15.to_csv('data/classification/validation15.csv', index=False)