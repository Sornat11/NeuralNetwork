# Analiza i porównanie wybranych architektur sieci neuronowych

**Analysis and Comparison of Selected Neural Network Architectures for Regression, Classification, and Image Recognition Tasks**

Projekt akademicki porównujący różne architektury sieci neuronowych (MLP, CNN, RNN/LSTM/GRU) w trzech typach zadań.

## Autorzy

- **Jakub Sornat**
- **Maciej Tajs**
- **Bartłomiej Sadza**

**Kierunek:** Informatyka i Ekonometria
**Prowadzący:** dr inż. Radosław Puka
**Rok:** 2025

---

## 📋 Spis treści

- [O projekcie](#o-projekcie)
- [Funkcjonalności](#funkcjonalności)
- [Struktura projektu](#struktura-projektu)
- [Instalacja](#instalacja)
- [Użycie](#użycie)
- [Eksperymenty](#eksperymenty)
- [Wyniki](#wyniki)
- [Technologie](#technologie)

---

## O projekcie

Projekt realizuje kompleksową analizę i porównanie wybranych architektur sieci neuronowych dla trzech rodzajów problemów:

1. **Problem klasyfikacyjny** - Iris, Wine (własny MLP + Keras MLP)
2. **Problem regresyjny** - Airline Passengers szereg czasowy (MLP + CNN + LSTM/GRU)
3. **Problem analizy obrazów** - Fashion MNIST (własny MLP + Keras MLP + CNN)

### Kluczowe cechy:

- ✅ **Własna implementacja MLP** od zera (NumPy) z backpropagation
- ✅ **Optimizery**: SGD, SGD+Momentum, Adam, RMSprop
- ✅ **Gotowe modele**: CNN, RNN/LSTM/GRU w Keras/TensorFlow
- ✅ **Framework eksperymentalny** z automatycznym grid search
- ✅ **Wielokrotne powtórzenia** (min. 5x) dla każdego zestawu parametrów
- ✅ **Metryki**: accuracy, precision, recall, F1, MSE, MAE, R²
- ✅ **Wizualizacje**: learning curves, confusion matrix, porównania
- ✅ **Podział danych**: 80/20 i 70/15/15

---

## Funkcjonalności

### 1. Własna implementacja MLP

```python
from src.manual_mlp.model import Model
from src.manual_mlp.layers import LayerDense
from src.manual_mlp.activations import ActivationReLU
from src.manual_mlp.losses import SoftmaxCategoricalCrossentropy
from src.manual_mlp.optimizers import OptimizerAdam

# Zbuduj model
model = Model()
model.add(LayerDense(4, 64))
model.add(ActivationReLU())
model.add(LayerDense(64, 3))

# Skonfiguruj
model.set(
    loss=SoftmaxCategoricalCrossentropy(),
    optimizer=OptimizerAdam(learning_rate=0.001)
)

# Trenuj
history = model.fit(X_train, y_train, epochs=100, batch_size=32)
```

### 2. Framework eksperymentalny

```python
from experiments.experiment_runner import ExperimentRunner, create_param_grid

runner = ExperimentRunner(results_dir="results")

param_grid = create_param_grid(
    base_params={"epochs": 100, "batch_size": 32},
    variations={
        "n_layers": [1, 2, 3],
        "neurons": [32, 64, 128],
        "learning_rate": [0.001, 0.01, 0.1],
        "optimizer": ["sgd", "adam", "rmsprop"],
    }
)

results = runner.run_experiment(
    experiment_name="iris_classification",
    model_fn=create_model,
    train_fn=train_model,
    eval_fn=evaluate_model,
    data=dataset,
    param_grid=param_grid,
    n_repeats=5,  # Wielokrotne powtórzenia
)
```

### 3. Wizualizacje

```python
from experiments.visualizations import (
    plot_learning_curves,
    plot_confusion_matrix,
    plot_parameter_comparison,
    plot_model_comparison
)

# Learning curves
plot_learning_curves(history, save_path="results/learning_curves.png")

# Porównanie parametrów
plot_parameter_comparison(
    results_df,
    param_name="learning_rate",
    metrics=["test_accuracy_mean", "test_f1_score_mean"]
)

# Porównanie modeli
plot_model_comparison(
    {"Custom MLP": results1, "Keras MLP": results2, "CNN": results3},
    metric="test_accuracy_mean"
)
```

---

## Struktura projektu

```
NeuralNetwork/
│
├── data/                           # Moduły do ładowania danych
│   ├── datasets.py                 # Ładowanie Iris, Wine, Fashion MNIST, Airline
│   ├── preprocessing.py            # Preprocessing
│   └── sample_data_generator.py    # Generator danych syntetycznych
│
├── src/
│   ├── manual_mlp/                 # WŁASNA IMPLEMENTACJA MLP (NumPy)
│   │   ├── model.py                # Model z forward/backward pass
│   │   ├── layers.py               # LayerDense z backpropagation
│   │   ├── activations.py          # ReLU, Softmax, Sigmoid, Linear
│   │   ├── losses.py               # Categorical CE, MSE, MAE
│   │   ├── optimizers.py           # SGD, Adam, RMSprop
│   │   └── metrics.py              # Accuracy, Precision, Recall, F1, R²
│   │
│   └── models/                     # MODELE W KERAS/TENSORFLOW
│       ├── multilayer_perceptron.py  # MLP w Keras
│       ├── convolutional_nn.py      # CNN w Keras
│       └── recurrent_nn.py          # RNN/LSTM/GRU w Keras
│
├── experiments/                    # Framework eksperymentalny
│   ├── experiment_runner.py        # Runner z grid search i powtórzeniami
│   ├── visualizations.py           # Wizualizacje wyników
│   ├── run_classification_experiments.py    # Eksperymenty klasyfikacyjne
│   ├── run_regression_experiments.py        # Eksperymenty regresyjne
│   └── run_image_experiments.py             # Eksperymenty na obrazach
│
├── results/                        # Wyniki eksperymentów (CSV, JSON, PNG)
│
├── utils/
│   └── seed.py                     # Ustawianie seed dla reproducibility
│
├── requirements.txt                # Zależności
├── main.py                         # Główny punkt wejścia
└── README.md                       # Ten plik
```

---

## Instalacja

### 1. Sklonuj repozytorium

```bash
git clone https://github.com/your-username/NeuralNetwork.git
cd NeuralNetwork
```

### 2. Utwórz środowisko wirtualne (zalecane)

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Zainstaluj zależności

```bash
pip install -r requirements.txt
```

**Wymagane biblioteki:**
- numpy
- pandas
- tensorflow (>=2.10)
- scikit-learn
- matplotlib
- seaborn
- optuna (opcjonalnie)

---

## Użycie

### Szybki start

```bash
# Uruchom interaktywne menu
python main.py
```

### Uruchamianie eksperymentów

#### 1. Eksperymenty klasyfikacyjne (Iris + Wine)

```bash
cd experiments
python run_classification_experiments.py
```

Testuje:
- Własny MLP vs Keras MLP
- Parametry: liczba warstw, neurony, learning rate, optimizers, momentum
- 5 powtórzeń każdego zestawu parametrów
- Wyniki zapisywane do `results/`

#### 2. Eksperymenty regresyjne (Airline Passengers)

```bash
cd experiments
python run_regression_experiments.py
```

Testuje:
- MLP vs LSTM vs GRU
- Szeregi czasowe (lookback=12)
- Parametry: warstwy, jednostki, learning rate
- Metryki: MSE, MAE, R²

#### 3. Eksperymenty na obrazach (Fashion MNIST)

```bash
cd experiments
python run_image_experiments.py
```

Testuje:
- Własny MLP vs Keras MLP vs CNN
- 10 klas ubrań (28x28 pikseli)
- Confusion matrix dla najlepszego modelu
- Porównanie accuracy

---

## Eksperymenty

### Parametry testowane w projekcie

Zgodnie z wymaganiami projektu, testujemy **minimum 4 wartości każdego parametru**:

| Parametr | Wartości testowane |
|----------|-------------------|
| **Liczba warstw** | 1, 2, 3, 4 |
| **Liczba neuronów** | 32, 64, 128, 256 |
| **Learning rate** | 0.0001, 0.001, 0.01, 0.1 |
| **Optimizer** | SGD, SGD+Momentum, Adam, RMSprop |
| **Momentum** | 0.0, 0.5, 0.9, 0.99 |

### Wielokrotne powtórzenia

Każdy zestaw parametrów jest trenowany **minimum 5 razy** (zgodnie z wymaganiami), ponieważ uczenie sieci nie jest deterministyczne.

### Metryki

**Klasyfikacja:**
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- Confusion matrix

**Regresja:**
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

### Zbiory danych

Każdy eksperyment ewaluuje na **trzech zbiorach** (zgodnie z wymaganiami):
- **Train set** - dane treningowe
- **Validation set** - dane walidacyjne (do strojenia)
- **Test set** - dane testowe (do końcowej ewaluacji)

---

## Wyniki

Wyniki są automatycznie zapisywane w katalogu `results/`:

```
results/
├── custom_mlp_iris_20250106_143022.csv      # Wyniki w CSV
├── custom_mlp_iris_20250106_143022.json     # Wyniki w JSON
├── keras_mlp_iris_20250106_143530.csv
├── iris_model_comparison.png                # Wykresy porównawcze
├── iris_custom_learning_rate.png
├── fashion_mnist_confusion_matrix.png
└── ...
```

### Format wyników CSV

Kolumny zawierają:
- Parametry modelu (n_layers, neurons, learning_rate, optimizer, momentum)
- Metryki dla **train/val/test** z:
  - `_mean` - średnia z 5+ powtórzeń
  - `_std` - odchylenie standardowe
  - `_min` - minimalna wartość
  - `_max` - maksymalna wartość
  - `_best` - najlepsza wartość

Przykład:
```csv
n_layers,neurons,learning_rate,optimizer,test_accuracy_mean,test_accuracy_std,test_accuracy_best
2,64,0.001,adam,0.9533,0.0123,0.9667
2,128,0.001,adam,0.9600,0.0089,0.9733
...
```

---

## Technologie

### Własna implementacja (NumPy)

- **Forward propagation** - przejście sygnału przez sieć
- **Backpropagation** - obliczanie gradientów
- **Optimizery**:
  - SGD (Stochastic Gradient Descent)
  - SGD + Momentum
  - Adam (Adaptive Moment Estimation)
  - RMSprop (Root Mean Square Propagation)
- **Funkcje aktywacji**: ReLU, Softmax, Sigmoid, Linear
- **Funkcje straty**: Categorical Crossentropy, MSE, MAE

### Gotowe modele (Keras/TensorFlow)

- **MLP** - Multilayer Perceptron
- **CNN** - Convolutional Neural Network (Conv2D + Pooling)
- **RNN** - Recurrent Neural Network (LSTM, GRU)
- **Conv1D-LSTM Hybrid** - dla szeregów czasowych

### Narzędzia

- **scikit-learn** - podział danych, metryki, datasety (Iris, Wine)
- **matplotlib + seaborn** - wizualizacje
- **pandas** - zarządzanie wynikami
- **optuna** (opcjonalnie) - automatyczna optymalizacja hiperparametrów

---

## Reprodukowalność

Wszystkie eksperymenty używają `set_seed()` do zapewnienia reprodukowalności:

```python
from utils.seed import set_seed
set_seed(42)  # Ten sam seed = te same wyniki
```

---

## Sprawozdanie

Pełne sprawozdanie projektu dostępne jest tutaj:

📄 [Project Report (DOCX)](https://aghedupl-my.sharepoint.com/:w:/r/personal/jakubsornat_student_agh_edu_pl/Documents/report.docx?d=w719a3c159b694350a6cdfea27e91fec0&csf=1&web=1&e=UyxR3n)

---

## Licencja

Projekt edukacyjny / Educational purposes only

---

## Kontakt

W razie pytań skontaktuj się z autorami:
- Jakub Sornat
- Maciej Tajs
- Bartłomiej Sadza

---

**Projekt wykonany w ramach kursu "Sieci Neuronowe i Uczenie Głębokie"**
**AGH Kraków, 2025**
