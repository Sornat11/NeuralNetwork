import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from correlation import smart_corr_heatmap

def detect_outliers_iqr(df, columns):
    outlier_summary = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask = (df[col] < lower) | (df[col] > upper)
        outlier_summary[col] = {
            "outlier_count": mask.sum(),
            "outlier_pct": round(100 * mask.sum() / len(df), 2),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2)
        }
    return pd.DataFrame(outlier_summary).T


performance_data = pd.read_csv('data/regression_our/student_preformance.csv')
print(performance_data.head())
print(performance_data.info())
print(performance_data.isna().sum()/len(performance_data)*100)
# mało braków, usuwamy je
performance_data = performance_data.dropna()

numerical_cols = [
    'Hours_Studied',
    'Attendance',
    'Sleep_Hours',
    'Previous_Scores',
    'Tutoring_Sessions',
    'Physical_Activity',
    'Exam_Score'
]

categorical_cols = [
    'Extracurricular_Activities',  # Yes/No
    'Internet_Access',             # Yes/No
    'Learning_Disabilities',       # Yes/No
    'School_Type',                 # Public/Private
    'Gender'                       # Male/Female
]

ordinal_cols = [
    'Parental_Involvement',     # Low, Medium, High
    'Access_to_Resources',      # Low, Medium, High
    'Motivation_Level',         # Low, Medium, High
    'Family_Income',            # Low, Medium, High
    'Teacher_Quality',          # Low, Medium, High
    'Peer_Influence',           # Negative, Neutral, Positive
    'Parental_Education_Level', # High School, College, Postgraduate
    'Distance_from_Home'        # Near, Moderate, Far
]

target = 'Exam_Score'

for col in performance_data.columns:
    if col in categorical_cols or col in ordinal_cols:
        print(performance_data[col].unique())

ordinal_maps = {
    'Parental_Involvement': {'Low': 1, 'Medium': 2, 'High': 3},
    'Access_to_Resources': {'Low': 1, 'Medium': 2, 'High': 3},
    'Motivation_Level': {'Low': 1, 'Medium': 2, 'High': 3},
    'Family_Income': {'Low': 1, 'Medium': 2, 'High': 3},
    'Teacher_Quality': {'Low': 1, 'Medium': 2, 'High': 3},
    'Peer_Influence': {'Negative': 1, 'Neutral': 2, 'Positive': 3},
    'Parental_Education_Level': {'High School': 1, 'College': 2, 'Postgraduate': 3},
    'Distance_from_Home': {'Near': 1, 'Moderate': 2, 'Far': 3}
}

for col, mapping in ordinal_maps.items():
    performance_data[col] = performance_data[col].map(mapping)

binary_maps = {
    'Extracurricular_Activities': {'No': 0, 'Yes': 1},
    'Internet_Access': {'No': 0, 'Yes': 1},
    'Learning_Disabilities': {'No': 0, 'Yes': 1},
    'School_Type': {'Public': 0, 'Private': 1},
    'Gender': {'Female': 0, 'Male': 1}
}

for col, mapping in binary_maps.items():
    if col in performance_data.columns:
        performance_data[col] = (
            performance_data[col]
            .astype(str)
            .str.strip()
            .str.title()  
            .map(mapping)
        )

print(performance_data.head())

# corr_df = smart_corr_heatmap(
#     performance_data,
#     numerical_cols=numerical_cols,
#     ordinal_cols=ordinal_cols,
#     categorical_cols=categorical_cols,
#     figsize=(14, 10)
# )

# Zmienne objaśniane nie mają korelacji między sobą
# większośc zmiennych nie ma silnej korelacji z targetem, pozostawiono te,
# które mają ją na moduł większą niż 0.15, co finalnie daje 5 zmiennych objaśniających
# 3 są numeryczne, 2 porządkowe

performance_data = performance_data[['Exam_Score','Access_to_Resources','Previous_Scores','Hours_Studied','Parental_Involvement','Attendance']]
print(performance_data.head())

numeric_cols = ['Exam_Score', 'Previous_Scores', 'Hours_Studied', 'Attendance']
outlier_report = detect_outliers_iqr(performance_data, numeric_cols)
print(outlier_report)

# outliery git. Nie usuwamy ich, bo mogą być istotne w kontekście wyników egzaminów.
# === 1) Podział na cechy i target ===
X = performance_data.drop(columns=['Exam_Score'])
y = performance_data['Exam_Score']

# === 2) Podział na zbiory ===
# Wariant 1: 80/20
X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Wariant 2: 70/15/15
X_temp, X_test_15, y_temp, y_test_15 = train_test_split(
    X, y, test_size=0.15, random_state=42
)
X_train_70, X_val_15, y_train_70, y_val_15 = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42  # ~15% z całości
)

# === 3) Standaryzacja TYLKO numerycznych kolumn objaśniających ===
num_cols = ['Previous_Scores', 'Hours_Studied', 'Attendance']

scaler = StandardScaler()

# dopasowujemy na train, transformujemy wszystko
X_train_80[num_cols] = scaler.fit_transform(X_train_80[num_cols])
X_test_20[num_cols] = scaler.transform(X_test_20[num_cols])

X_train_70[num_cols] = scaler.fit_transform(X_train_70[num_cols])
X_val_15[num_cols] = scaler.transform(X_val_15[num_cols])
X_test_15[num_cols] = scaler.transform(X_test_15[num_cols])

# === 4) Zapis do plików ===
X_train_80.assign(Exam_Score=y_train_80).to_csv('data/regression_our/train80.csv', index=False)
X_test_20.assign(Exam_Score=y_test_20).to_csv('data/regression_our/test20.csv', index=False)
X_train_70.assign(Exam_Score=y_train_70).to_csv('data/regression_our/train70.csv', index=False)
X_val_15.assign(Exam_Score=y_val_15).to_csv('data/regression_our/validation15.csv', index=False)
X_test_15.assign(Exam_Score=y_test_15).to_csv('data/regression_our/test15.csv', index=False)

