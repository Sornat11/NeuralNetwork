
# Neural Network Project – Przewodnik naukowy

**Projekt naukowy realizowany w ramach kursu/pracy dyplomowej.**

Framework do eksperymentów z sieciami neuronowymi (klasyfikacja, regresja, obrazy):
- własna implementacja MLP (NumPy)
- modele Keras (MLP, CNN, CNN-1D, LSTM)
- automatyzacja eksperymentów i wizualizacji
- eksport wyników do Excela

Projekt spełnia wytyczne akademickie (patrz: `wytyczne_do_projektu.pdf`) i jest gotowy do rozbudowy o nowe architektury, zbiory danych i metody analizy.

---

## Spis treści
1. [Cel projektu](#cel-projektu)
2. [Funkcjonalności](#funkcjonalności)
3. [Struktura katalogów](#struktura-katalogów)
4. [Jak zacząć](#jak-zacząć)
5. [Uruchamianie eksperymentów](#uruchamianie-eksperymentów)
6. [Wizualizacje](#wizualizacje)
7. [Co dopisać w raporcie](#co-dopisać-w-raporcie)
8. [Szacowany czas pracy](#szacowany-czas-pracy)
9. [Troubleshooting](#troubleshooting)
10. [Testy jednostkowe (opcjonalnie)](#testy-jednostkowe)
11. [Autorzy](#autorzy)

---

## Cel projektu

Projekt powstał jako praca naukowa mająca na celu:
- porównanie własnej implementacji sieci neuronowych z rozwiązaniami Keras,
- analizę wpływu hiperparametrów na wyniki,
- automatyzację eksperymentów i raportowania,
- wyciągnięcie wniosków przydatnych w dalszych badaniach nad uczeniem maszynowym.

---

## Funkcjonalności
- Ręczna implementacja MLP (NumPy)
- Modele Keras: MLP, CNN 2D, CNN 1D, LSTM
- Automatyczny grid search po hiperparametrach
- Eksperymenty na 5 zbiorach danych (tabularne i obrazy)
- Eksport wyników do Excela
- Generowanie wykresów (learning curves, confusion matrix, scatter)
- Szczegółowa instrukcja uruchomienia
- Testy jednostkowe (opcjonalnie)

---

## Struktura katalogów
```
NeuralNetwork/
├── main.py, main_keras.py, main_fashion_mnist.py, main_regression_advanced.py
├── run_all_experiments.py
├── requirements.txt
├── src/
│   ├── manual_mlp/         # Ręczna implementacja MLP
│   └── models/             # Modele Keras: MLP, CNN, LSTM
├── utils/
│   ├── experiment_runner.py
│   ├── keras_experiment_runner.py
│   └── visualization.py
├── data/                   # Zbiory danych i preprocessing
├── results/                # Wyniki (Excel, wykresy)
│   └── visualizations/
├── report/
│   ├── raport.tex, literatura_template.md
│   └── Makefile
├── tests/                  # Testy jednostkowe (opcjonalnie)
├── README.md, JAK_URUCHOMIC.md, PODSUMOWANIE_PRACY.md, INSTRUKCJA_UZUPELNIENIA.md
├── wytyczne_do_projektu.pdf
```

---

## Jak zacząć

### Środowisko wirtualne (zalecane)

Środowisko wirtualne pozwala odizolować zależności projektu od reszty systemu i uniknąć konfliktów między różnymi projektami Python. Dzięki temu masz pewność, że wszystkie pakiety są zgodne z wymaganiami projektu.

**Tworzenie i aktywacja środowiska:**

Na Windows:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```
Na Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

**Dezaktywacja środowiska:**
```bash
deactivate
```

---

1. Utwórz i aktywuj środowisko wirtualne (patrz wyżej).
2. Zainstaluj zależności:
	```bash
	pip install -r requirements.txt
	```
3. Uruchom wszystkie eksperymenty (6–10h):
	```bash
	python run_all_experiments.py
	```

---

## Uruchamianie eksperymentów

1. **Manual MLP (jeśli nie uruchamiałeś wcześniej):**
	 ```bash
	 python main.py
	 ```
2. **Keras MLP:**
	 ```bash
	 python main_keras.py
	 ```
3. **Fashion MNIST (MLP + CNN):**
	 ```bash
	 python main_fashion_mnist.py
	 ```
4. **Zaawansowane regresje (CNN-1D + LSTM):**
	 ```bash
	 python main_regression_advanced.py
	 ```

Wynik: ~21 plików Excel w `results/`.

---

## Wizualizacje

- **Podstawowe (Manual + Keras MLP):**
	```bash
	python generate_visualizations.py
	```
- **Rozszerzone (CNN, LSTM, Fashion MNIST):**
	```bash
	python generate_visualizations_extended.py
	```

Wynik: ~40–46 plików PNG w `results/visualizations/`.

---

## Troubleshooting

- **Brak pakietów:**
	```bash
	pip install -r requirements.txt
	```
- **Eksperymenty za wolne:**
	Zmniejsz gridy:
	```python
	HIDDEN_LAYERS_GRID = [2, 3]
	NEURONS_GRID = [16, 32]
	```
- **Brak pamięci RAM:**
	Zmniejsz batch_size w odpowiednich plikach.
- **TensorFlow warnings:**
	Można ignorować ostrzeżenia o optymalizacji binariów.
- **Wizualizacje nie wyświetlają się:**
	Sprawdź ścieżki do plików PNG.

---

## Testy jednostkowe (opcjonalnie)
```bash
pip install pytest
pytest tests/
```

---

## Autorzy

- Jakub Sornat
- Maciej Tajs
- Bartłomiej Sadza

---

**Projekt spełnia wytyczne naukowe i jest gotowy do dalszych badań! Powodzenia! 🚀**