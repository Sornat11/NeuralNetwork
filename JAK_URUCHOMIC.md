# 🚀 JAK URUCHOMIĆ WSZYSTKIE EKSPERYMENTY

## Szybki Start (wszystko naraz)

```bash
# Uruchom WSZYSTKIE eksperymenty (6-10h)
python3 run_all_experiments.py
```

**UWAGA:** To może trwać **6-10 godzin**! Lepiej uruchomić na noc lub weekend.

---

## Uruchamianie krok po kroku (rekomendowane)

Jeśli chcesz kontrolować proces lub masz mało czasu:

### KROK 1: Eksperymenty Manual MLP (już zrobione?)

```bash
# Jeśli NIE uruchamiałeś wcześniej:
python3 main.py

# Czas: ~2-3h
# Generuje: 8 plików Excel w results/
```

**Sprawdź czy masz:**
- `results/manual_perceptron_wyniki_classification_train_test.xlsx`
- `results/manual_perceptron_wyniki_classification_train_val_test.xlsx`
- ... (6 więcej plików)

Jeśli TAK - możesz pominąć ten krok! ✅

---

### KROK 2: Eksperymenty Keras MLP

```bash
python3 main_keras.py

# Czas: ~2-3h
# Generuje: 8 plików Excel z wynikami Keras MLP
```

---

### KROK 3: Fashion MNIST (MLP + CNN)

```bash
python3 main_fashion_mnist.py

# Czas: ~1.5-2h
# Generuje: 3 pliki Excel
#   - manual_perceptron_wyniki_fashion_mnist.xlsx
#   - keras_mlp_wyniki_fashion_mnist.xlsx
#   - keras_cnn_wyniki_fashion_mnist.xlsx
```

---

### KROK 4: Zaawansowane modele regresji (CNN 1D + LSTM)

```bash
python3 main_regression_advanced.py

# Czas: ~1-1.5h
# Generuje: 2 pliki Excel
#   - keras_cnn1d_wyniki_regression.xlsx
#   - keras_lstm_wyniki_regression.xlsx
```

---

## Po zakończeniu eksperymentów

### KROK 5: Generowanie wizualizacji

```bash
# Podstawowe wizualizacje (dla manual i keras MLP)
python3 generate_visualizations.py

# Rozszerzone wizualizacje (CNN, LSTM, Fashion MNIST)
python3 generate_visualizations_extended.py

# Czas: ~30-45 min
# Generuje: ~40+ wykresów PNG w results/visualizations/
```

---

### KROK 6: Generowanie tabel LaTeX dla raportu

```bash
python3 utils/latex_generator.py

# Czas: ~1 min
# Generuje: report/wyniki_generated.tex
```

---

### KROK 7: Uzupełnienie raportu

1. Otwórz `report/raport.tex`
2. W sekcji "Wyniki i analiza" (sekcja 6):
   - Wklej zawartość z `report/wyniki_generated.tex`
3. Napisz analizę wyników (2-3h):
   - Interpretacja wykresów
   - Porównanie modeli
   - Wpływ hiperparametrów
4. Napisz wnioski (sekcja 7, 1-2h):
   - Co się udało
   - Problemy napotkane
   - Możliwe usprawnienia

---

### KROK 8: Kompilacja raportu PDF

```bash
cd report
make

# Lub bez make:
pdflatex raport.tex
pdflatex raport.tex  # Dwa razy dla TOC

# Otwórz PDF:
make open
# lub: open raport.pdf
```

---

## Troubleshooting

### Błąd: "ModuleNotFoundError"

```bash
pip install -r requirements.txt
```

### Eksperymenty trwają za długo

Możesz zmniejszyć grid search:
1. Otwórz `main_fashion_mnist.py` lub `main_regression_advanced.py`
2. Zmniejsz listy hiperparametrów, np.:
   ```python
   HIDDEN_LAYERS_GRID = [2, 3]  # Zamiast [2, 3, 4]
   NEURONS_GRID = [64, 128]     # Zamiast [64, 128, 256]
   ```

### Brak pamięci RAM podczas treningu

Zmniejsz batch_size:
```python
batch_size=64  # Zamiast 128
```

### TensorFlow warnings

Ignoruj ostrzeżenia typu "This TensorFlow binary is optimized..." - to nie wpływa na wyniki.

---

## Struktura wyników (po wszystkich eksperymentach)

```
results/
├── manual_perceptron_wyniki_*.xlsx (8 plików)
├── keras_mlp_wyniki_*.xlsx (8 plików)
├── keras_cnn_wyniki_fashion_mnist.xlsx
├── keras_cnn1d_wyniki_regression.xlsx
├── keras_lstm_wyniki_regression.xlsx
└── visualizations/
    ├── *_learning_curves.png (~20 plików)
    ├── *_confusion_matrix.png (~8 plików)
    ├── *_scatter.png (~8 plików)
    └── *_comparison.png (~10 plików)
```

**RAZEM: ~21 plików Excel + 46+ wykresów PNG**

---

## Harmonogram rekomendowany

### Dzień 1 (sobota):
```
09:00-11:00  → main.py (manual MLP)
11:00-13:00  → main_keras.py (Keras MLP)
13:00-14:00  → Przerwa
14:00-16:00  → main_fashion_mnist.py
16:00-17:00  → main_regression_advanced.py
17:00-18:00  → Generowanie wizualizacji
```

### Dzień 2 (niedziela):
```
10:00-13:00  → Pisanie analizy wyników
13:00-14:00  → Przerwa
14:00-17:00  → Pisanie wniosków + kompilacja PDF
17:00-18:00  → Przegląd finalny
```

---

## Pytania?

Jeśli coś nie działa:
1. Sprawdź logi błędów
2. Zobacz komentarze w kodzie
3. Sprawdź czy masz wszystkie zależności: `pip list`

**Powodzenia! 🎉**
