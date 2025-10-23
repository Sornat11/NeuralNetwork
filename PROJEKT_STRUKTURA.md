# Propozycja struktury projektu

```
NeuralNetwork/
│
├── main.py                  # Główny plik uruchomieniowy, interfejs CLI lub config
├── README.md                # Dokumentacja projektu
├── requirements.txt         # Lista zależności
├── TODO.md                  # Lista zadań
│
├── data/                    # Przetwarzanie i przykładowe dane
│   ├── classification.csv   # Dane do klasyfikacji
│   ├── regression.csv       # Dane do regresji
│   ├── preprocessing.py     # Skrypty do przetwarzania danych
│   └── xor.py               # Przykładowy generator danych XOR
│
├── nn/                      # Moduły sieci neuronowych
│   ├── __init__.py
│   ├── activationFunctions.py # Funkcje aktywacji
│   ├── formatting.py          # Formatowanie danych/wejść
│   ├── layers.py              # Warstwy sieci
│   ├── losses.py              # Funkcje straty
│   ├── mlp.py                 # Implementacja MLP (Multilayer Perceptron)
│   ├── rnn.py                 # Implementacja RNN (Recurrent Neural Network)
│   ├── cnn.py                 # Implementacja CNN (Convolutional Neural Network)
│   ├── optim.py               # Optymalizatory
│   └── utils.py               # Funkcje pomocnicze
│
├── tests/                   # Testy jednostkowe
│   ├── test_mlp.py           # Testy MLP
│   ├── test_rnn.py           # Testy RNN
│   ├── test_cnn.py           # Testy CNN
│   └── ...
│
└── .gitignore               # Plik ignorujący pliki/foldery w repozytorium
```

## Opis
- Każdy typ sieci (MLP, RNN, CNN) ma osobny moduł.
- Dane i przetwarzanie danych są wydzielone do osobnego folderu.
- Testy jednostkowe dla każdego modułu.
- Plik `main.py` jako punkt wejścia do projektu.
- Plik `README.md` z opisem i instrukcją uruchomienia.
- Plik `TODO.md` do śledzenia postępu.
