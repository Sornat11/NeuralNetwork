import pandas as pd
import numpy as np
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt

from correlation import smart_correlation_matrix

# === 0) Wczytanie i porządek w nazwach kolumn ===
adult_data = pd.read_csv('data/classification.csv')

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

adult_balanced.to_csv('data/adult_preprocessed.csv', index=False)