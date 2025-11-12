import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

def cramers_v(x, y):
    """Cramér’s V (bias corrected) — dla dwóch zmiennych kategorycznych."""
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

def correlation_ratio(categories, values):
    """
    Correlation Ratio (eta) — dla (kategoria, liczba).
    Poprawiona wersja: działa na ndarray, odporna na NaN i kategorie bez obserwacji.
    """
    # kategorie → kody (NaN dostaje -1)
    cats, _ = pd.factorize(categories)

    # wartości numeryczne jako ndarray (coerce dla safety)
    vals = pd.to_numeric(values, errors='coerce').to_numpy()

    # tylko obserwacje z poprawnymi wartościami
    valid = (~np.isnan(vals)) & (cats >= 0)
    vals = vals[valid]
    cats = cats[valid]

    if vals.size == 0:
        return np.nan

    n_cats = cats.max() + 1
    counts = np.bincount(cats, minlength=n_cats)

    # średnie w każdej kategorii
    means = np.empty(n_cats, dtype=float)
    for i in range(n_cats):
        v = vals[cats == i]
        means[i] = np.nan if v.size == 0 else np.nanmean(v)

    y_mean = np.nanmean(vals)
    numerator = np.nansum(counts * (means - y_mean) ** 2)
    denominator = np.nansum((vals - y_mean) ** 2)

    return 0.0 if denominator == 0 else np.sqrt(numerator / denominator)

def smart_correlation_matrix(df, plot=True, figsize=(16, 12), cmap='coolwarm'):
    """
    Tworzy macierz korelacji z odpowiednimi wskaźnikami:
    - num/num: Pearson
    - cat/cat: Cramér’s V
    - cat/num: Correlation Ratio (η)
    """
    df = df.copy()
    cols = df.columns
    n = len(cols)

    corr = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)

    # typy kolumn
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(
        df.select_dtypes(include=['bool']).columns, errors='ignore'
    )
    cat_cols = [c for c in df.columns if c not in num_cols]

    for i in range(n):
        for j in range(i, n):
            col1, col2 = cols[i], cols[j]
            x, y = df[col1], df[col2]

            if col1 == col2:
                corr.loc[col1, col2] = 1.0
                continue

            # num-num → Pearson
            if col1 in num_cols and col2 in num_cols:
                val = x.corr(y, method='pearson')

            # cat-cat → Cramér’s V
            elif col1 in cat_cols and col2 in cat_cols:
                val = cramers_v(x, y)

            # cat-num lub num-cat → Correlation Ratio (eta)
            elif col1 in cat_cols and col2 in num_cols:
                val = correlation_ratio(x, y)
            elif col1 in num_cols and col2 in cat_cols:
                val = correlation_ratio(y, x)
            else:
                val = np.nan

            corr.loc[col1, col2] = val
            corr.loc[col2, col1] = val

    # 👉 wartość bezwzględna dla wizualizacji
    corr_abs = corr.abs()

    if plot:
        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_abs,
            cmap=cmap,
            center=0,
            annot=True,          # ✅ wpisuje liczby
            fmt=".2f",            # format np. 0.73
            linewidths=0.3,
            cbar_kws={'shrink': 0.7},
            annot_kws={"size": 8}  # mniejszy font
        )
        plt.title("Smart mapa korelacji (|Pearson|, |Cramér’s V|, |η|)", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    return corr