import pandas as pd

adult_data = pd.read_csv('data/classification.csv')
adult_data.replace('?', pd.NA, inplace=True)
print(adult_data.isna().sum()/150.6)  # 15,060 rows in the dataset

# Są braki - w kolumnach workclass, occupation, native.country obczaimy czy można wywalić te wiersze.

cols_with_nans = ['workclass', 'occupation', 'native.country']

print("Cały zbiór:")
print(adult_data['income'].value_counts(normalize=True))

for col in cols_with_nans:
    mask = adult_data[col].isna()
    print(f"\nKolumna: {col}")
    print("\nWiersze z brakami:")
    print(adult_data.loc[mask, 'income'].value_counts(normalize=True))

# native.country ma podobny rozkład klasy docelowej, więc można usunąć wiersze z brakami
# workclass i occupation mają inny rozkład, więc usunięcie wierszy z brakami może zaburzyć rozkład klasy docelowej

adult_data.dropna(subset=['native.country'], inplace=True)

for col in adult_data.select_dtypes(include='object').columns:
    print(f"\nKolumna: {col}")
    print(adult_data[col].unique())

adult_data['occupation'] = adult_data['occupation'].fillna('Unknown')
adult_data['workclass'] = adult_data['workclass'].fillna('Unknown')

# tera ogarniamy klasy nieliczbowe
education_order = [
    'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th',
    'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm',
    'Bachelors', 'Masters', 'Prof-school', 'Doctorate'
]

adult_data['education'] = pd.Categorical(
    adult_data['education'],
    categories=education_order,
    ordered=True
)

adult_data['education'] = adult_data['education'].cat.codes
adult_data['sex'] = 1 if adult_data['sex'] == 'Male' else 0
adult_data['income'] = 1 if adult_data['income'] == '>50K' else 0
adult_data['native.country'] = adult_data['native.country'].apply(lambda x: 1 if x == 'United-States' else 0)




