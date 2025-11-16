# Instrukcja uzupełnienia projektu

Ten dokument opisuje kroki niezbędne do ukończenia projektu i wygenerowania raportu.

## Status projektu

### ✅ Co jest gotowe:

1. **Implementacja ręczna MLP** (`src/manual_mlp/`)
   - Model, warstwy, aktywacje, metryki
   - Pełna funkcjonalność

2. **Implementacja Keras** (`src/models/keras_mlp.py`)
   - Model MLP w Keras
   - Analogiczna architektura do ręcznej

3. **Experiment runners**
   - `utils/experiment_runner.py` - dla ręcznej implementacji
   - `utils/keras_experiment_runner.py` - dla Keras

4. **Preprocessing danych** (4 zbiory)
   - Classification, Classification_Our
   - Regression, Regression_Our

5. **Narzędzia wizualizacji** (`utils/visualization.py`)
   - Learning curves
   - Confusion matrices
   - Regression scatter plots
   - Wykresy porównawcze

6. **Struktura raportu LaTeX** (`report/raport.tex`)
   - Pełna struktura
   - Sekcje: wstęp, dane, preprocessing, architektura, metodologia, wyniki, wnioski

7. **Generator LaTeX** (`utils/latex_generator.py`)
   - Automatyczne generowanie tabel z wyników
   - Wstawianie wykresów

---

### ❌ Co trzeba zrobić:

## Krok 1: Uruchomienie eksperymentów Keras

**Czas: ~1.5-2h** (głównie czekanie na treningi)

```bash
# Uruchom eksperymenty Keras na wszystkich zbiorach
python main_keras.py
```

To wygeneruje 8 plików Excel w folderze `results/`:
- `keras_wyniki_classification_train_test.xlsx`
- `keras_wyniki_classification_train_val_test.xlsx`
- `keras_wyniki_classification_our_train_test.xlsx`
- `keras_wyniki_classification_our_train_val_test.xlsx`
- `keras_wyniki_regression_train_test.xlsx`
- `keras_wyniki_regression_train_val_test.xlsx`
- `keras_wyniki_regression_our_train_test.xlsx`
- `keras_wyniki_regression_our_train_val_test.xlsx`

**Uwaga:** Możesz przerwać w dowolnym momencie (Ctrl+C) i wznowić później.

---

## Krok 2: Generowanie wizualizacji

**Czas: ~15-20 min**

```bash
# Wygeneruj wszystkie wykresy
python generate_visualizations.py
```

To utworzy folder `results/visualizations/` z:
- Learning curves (dla manual i Keras, wszystkie zbiory)
- Confusion matrices (klasyfikacja)
- Regression scatter plots (regresja)
- Wykresy porównawcze manual vs Keras

---

## Krok 3: Generowanie fragmentów LaTeX

**Czas: ~1 min**

```bash
# Wygeneruj tabele i referencje do wykresów
python utils/latex_generator.py
```

To utworzy plik `report/wyniki_generated.tex` z gotowymi tabelami i wykresami.

---

## Krok 4: Uzupełnienie raportu

**Czas: ~2-4h** (pisanie analizy i wniosków)

### 4.1. Wstaw wygenerowane wyniki

Otwórz `report/raport.tex` i zastąp sekcję "Wyniki i analiza" zawartością z `report/wyniki_generated.tex`.

### 4.2. Napisz analizę wyników

W sekcji 6 ("Wyniki i analiza") dodaj:

- **Interpretację wykresów:**
  - Czy learning curves pokazują overfitting?
  - Jak szybko modele zbiegają?
  - Porównanie manual vs Keras

- **Analizę hiperparametrów:**
  - Jaki wpływ ma liczba warstw?
  - Jaki wpływ ma liczba neuronów?
  - Jaki learning rate działa najlepiej?

- **Obserwacje:**
  - Które zbiory były trudniejsze?
  - Dlaczego Keras jest lepszy/gorszy od manual?

### 4.3. Napisz wnioski

W sekcji 7 ("Wnioski") dodaj:

- **Porównanie implementacji:**
  - Różnice w dokładności
  - Różnice w czasie treningu
  - Co wyniósłeś z implementacji ręcznej?

- **Problemy napotkane:**
  - Trudności podczas implementacji
  - Jak je rozwiązaliście?

- **Możliwe usprawnienia:**
  - Co można poprawić?
  - Pomysły na rozszerzenie projektu

---

## Krok 5: Kompilacja raportu PDF

**Czas: ~1 min**

```bash
cd report
make
```

Lub jeśli nie masz `make`:

```bash
cd report
pdflatex raport.tex
pdflatex raport.tex  # Dwa razy dla TOC i referencji
```

To wygeneruje `report/raport.pdf`.

Otwórz i sprawdź:
```bash
make open
# lub
open raport.pdf
```

---

## Opcjonalne: Testy jednostkowe

**Czas: ~2-3h**

Jeśli chcecie pełną liczbę punktów, dodajcie testy:

```bash
# Zainstaluj pytest
pip install pytest

# Stwórz folder testów
mkdir tests

# Napisz testy (przykłady poniżej)
```

Przykładowe testy w `tests/test_layers.py`:

```python
import numpy as np
from src.manual_mlp.layers import LayerDense

def test_layer_forward():
    layer = LayerDense(2, 3)
    X = np.array([[1, 2]])
    output = layer.forward(X)
    assert output.shape == (1, 3)

def test_layer_backward():
    layer = LayerDense(2, 3)
    X = np.array([[1, 2]])
    output = layer.forward(X)
    dvalues = np.ones_like(output)
    dinputs = layer.backward(dvalues, X)
    assert dinputs.shape == X.shape
```

Uruchom testy:
```bash
pytest tests/
```

---

## Podsumowanie czasu

| Krok | Czas |
|------|------|
| 1. Eksperymenty Keras | 1.5-2h |
| 2. Generowanie wykresów | 15-20 min |
| 3. Generator LaTeX | 1 min |
| 4. Pisanie analizy | 2-4h |
| 5. Kompilacja PDF | 1 min |
| **RAZEM (minimum)** | **4-7h** |
| Opcjonalnie: Testy | +2-3h |

---

## Struktura plików (po wykonaniu)

```
NeuralNetwork/
├── results/
│   ├── manual_perceptron_wyniki_*.xlsx (8 plików)
│   ├── keras_wyniki_*.xlsx (8 plików)
│   └── visualizations/
│       ├── *_learning_curves.png
│       ├── *_confusion_matrix.png
│       ├── *_scatter.png
│       └── *_comparison.png
│
├── report/
│   ├── raport.tex
│   ├── raport.pdf ← KOŃCOWY RAPORT
│   ├── wyniki_generated.tex
│   └── Makefile
│
└── [reszta bez zmian]
```

---

## Troubleshooting

### Problem: `ModuleNotFoundError`

```bash
# Zainstaluj brakujące pakiety
pip install -r requirements.txt
```

### Problem: LaTeX nie kompiluje

Upewnij się, że masz zainstalowany LaTeX:
- **macOS:** `brew install mactex` lub pobierz z https://www.tug.org/mactex/
- **Linux:** `sudo apt-get install texlive-full`
- **Windows:** Pobierz MiKTeX z https://miktex.org/

### Problem: Wykresy nie wyświetlają się w PDF

Sprawdź, czy ścieżki do wykresów w `raport.tex` są poprawne (relatywne do folderu `report/`).

### Problem: Eksperymenty Keras trwają za długo

Możesz zredukować grid search:
- Zmniejsz liczbę epok (np. 30 zamiast 50)
- Zmniejsz liczbę runów (np. 1 zamiast 3)
- Zmniejsz grid (np. tylko [2, 3] warstwy zamiast [1,2,3,4])

Edytuj w `main_keras.py`:
```python
HIDDEN_LAYERS_GRID = [2, 3]  # Zamiast [1, 2, 3, 4]
NEURONS_GRID = [16, 32]      # Zamiast [8, 16, 32, 64]
```

---

## Pytania?

Jeśli coś nie działa:
1. Sprawdź logi błędów
2. Zobacz dokumentację w kodzie (docstringi)
3. Uruchom krok po kroku (nie wszystko naraz)

Powodzenia! 🚀