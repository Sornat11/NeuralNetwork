import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

def smart_corr_heatmap(
    df: pd.DataFrame,
    numerical_cols: list,
    ordinal_cols: list,
    categorical_cols: list,  # binarne
    figsize=(16, 12),
    cmap='coolwarm'
):
    # Zbiory typów
    num_set = set(numerical_cols)
    ord_set = set(ordinal_cols)
    bin_set = set(categorical_cols)

    # Kolumny istniejące w df
    cols = [c for c in (list(num_set | ord_set | bin_set)) if c in df.columns]

    # Lokalne helpery
    def to_binary_series(s: pd.Series) -> pd.Series:
        # sprowadź do 0/1; jeśli >2 poziomy -> NaN (brak sensu dla biserial)
        if pd.api.types.is_numeric_dtype(s):
            vals = pd.unique(s.dropna())
            if len(vals) == 2:
                ordered = np.sort(vals.astype(float))
                mapping = {ordered[0]: 0.0, ordered[1]: 1.0}
                return s.map(mapping)
        codes, _ = pd.factorize(s, sort=True)
        uniq = np.unique(codes[codes >= 0])
        if len(uniq) != 2:
            return pd.Series(np.nan, index=s.index)
        return pd.Series(codes, index=s.index).astype(float)

    def pair_corr(x: pd.Series, y: pd.Series, t1: str, t2: str) -> float:
        pair = pd.concat([x, y], axis=1).dropna()
        if pair.shape[0] < 3:
            return np.nan
        a, b = pair.iloc[:,0], pair.iloc[:,1]

        if t1 == 'num' and t2 == 'num':
            if a.std() == 0 or b.std() == 0: return np.nan
            return pearsonr(a, b)[0]

        if (t1 == 'ord' and t2 == 'ord') or (t1 == 'num' and t2 == 'ord') or (t1 == 'ord' and t2 == 'num'):
            if a.nunique() < 2 or b.nunique() < 2: return np.nan
            return spearmanr(a, b)[0]

        if t1 == 'bin' and t2 == 'bin':
            A = to_binary_series(a); B = to_binary_series(b)
            pair2 = pd.concat([A, B], axis=1).dropna()
            if pair2.shape[0] < 3: return np.nan
            A, B = pair2.iloc[:,0], pair2.iloc[:,1]
            if A.std() == 0 or B.std() == 0: return np.nan
            return pearsonr(A, B)[0]

        if (t1 == 'bin' and t2 in ('num','ord')) or (t2 == 'bin' and t1 in ('num','ord')):
            if t1 == 'bin':
                A = to_binary_series(a); B = b
            else:
                A = a; B = to_binary_series(b)
            pair2 = pd.concat([A, B], axis=1).dropna()
            if pair2.shape[0] < 3: return np.nan
            A, B = pair2.iloc[:,0], pair2.iloc[:,1]
            if A.std() == 0 or B.std() == 0: return np.nan
            return pearsonr(A, B)[0]

        return np.nan

    # Mapa typów
    col_type = {c: ('num' if c in num_set else 'ord' if c in ord_set else 'bin') for c in cols}

    # Liczenie macierzy korelacji
    corr = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i, c1 in enumerate(cols):
        corr.loc[c1, c1] = 1.0
        for j in range(i+1, len(cols)):
            c2 = cols[j]
            r = pair_corr(df[c1], df[c2], col_type[c1], col_type[c2])
            corr.loc[c1, c2] = r
            corr.loc[c2, c1] = r

    # Heatmapa z |r|
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr.abs(),
        cmap=cmap,
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={'shrink': 0.7},
        annot_kws={'size': 8}
    )
    plt.title("Heatmapa korelacji: Pearson / Spearman / Phi / point-biserial (|r|)")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return corr