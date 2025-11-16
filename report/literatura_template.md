# Przegląd literatury - Template

Ten dokument zawiera template/wytyczne do napisania przeglądu literatury dla każdego zbioru danych.

## Struktura przeglądu literatury (dla każdego zbioru)

### 1. Nazwa zbioru danych

**Opis:** [Krótki opis zbioru]

**Źródło:** [Link do źródła]

### 2. Przegląd prac wykorzystujących ten zbiór

#### Praca 1: [Tytuł]
- **Autorzy:** [Nazwiska autorów]
- **Rok:** [Rok publikacji]
- **Źródło:** [Link lub DOI]
- **Metody użyte:** [Np. SVM, Random Forest, Neural Networks]
- **Wyniki:** [Accuracy, MSE, etc. - konkretne liczby]
- **Preprocessing:** [Jakie techniki preprocessingu zastosowali]

#### Praca 2: [Tytuł]
[Analogicznie jak powyżej]

#### Praca 3: [Tytuł]
[Analogicznie jak powyżej]

### 3. Podsumowanie

**Najlepsze wyniki z literatury:**
- [Metoda X]: accuracy = XX%
- [Metoda Y]: accuracy = YY%

**Nasze wyniki (do porównania):**
- Manual MLP: accuracy = [Twój wynik]%
- Keras MLP: accuracy = [Twój wynik]%
- CNN: accuracy = [Twój wynik]% (jeśli dotyczy)

**Wnioski:**
- Czy nasze wyniki są porównywalne?
- Dlaczego mogą być lepsze/gorsze?

---

## JAK ZNALEŹĆ LITERATURĘ?

### 1. Google Scholar
https://scholar.google.com

**Przykładowe zapytania:**
- "Adult Income dataset machine learning"
- "Fashion MNIST classification"
- "Stock market prediction neural networks"
- "Loan approval prediction"

### 2. Papers With Code
https://paperswithcode.com

**Wyszukaj:**
- Nazwę datasetu
- Zobacz "State of the Art" wyniki
- Sprawdź jakie modele dały najlepsze wyniki

### 3. Kaggle
https://www.kaggle.com

**Wyszukaj:**
- Nazwę datasetu
- Zobacz "Notebooks" i "Leaderboard"
- Sprawdź jakie wyniki osiągnęli inni

### 4. UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/

**Dla Adult Income i innych zbiorów UCI:**
- Strona zbioru zawiera często linki do prac
- Sekcja "Relevant Papers"

---

## PRZYKŁAD DLA FASHION MNIST

### Fashion MNIST Dataset

**Opis:** Zbiór 70,000 obrazów odzieży (28x28 pikseli, 10 klas), alternatywa dla MNIST.

**Źródło:** Zalando Research (https://github.com/zalandoresearch/fashion-mnist)

### Przegląd literatury

#### Praca 1: Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms
- **Autorzy:** Han Xiao, Kashif Rasul, Roland Vollgraf
- **Rok:** 2017
- **Źródło:** https://arxiv.org/abs/1708.07747
- **Metody użyte:**
  - SVM: 89.7%
  - Random Forest: 87.6%
  - Simple CNN: 92.3%
  - ResNet: 94.9%
- **Wyniki:** CNN osiąga accuracy ~92-95%
- **Preprocessing:** Normalizacja do [0,1], data augmentation

#### Praca 2: Deep Learning Performance on Fashion-MNIST
- **Autorzy:** [Przykładowi autorzy]
- **Rok:** 2019
- **Metody użyte:**
  - MLP (3 warstwy): 88.4%
  - CNN (5 warstw): 93.1%
  - VGG-like: 94.2%
- **Wyniki:** Głębokie CNN (8+ warstw) osiągają >94%

### Podsumowanie Fashion MNIST

**Najlepsze wyniki z literatury:**
- ResNet: 94.9%
- CNN (deep): ~94%
- MLP: ~88-89%

**Nasze wyniki:**
- Manual MLP: [TODO: wpisać po eksperymentach]
- Keras MLP: [TODO: wpisać po eksperymentach]
- Keras CNN: [TODO: wpisać po eksperymentach]

**Wnioski:**
- MLP typowo osiąga 87-89% na Fashion MNIST
- CNN osiąga 92-95%
- Nasze wyniki powinny być w tych zakresach
- Jeśli niższe - trzeba przeanalizować dlaczego (np. za mały grid, za mało epok)

---

## PRZYKŁAD DLA ADULT INCOME

### Adult Income Dataset (UCI)

**Opis:** Przewidywanie czy osoba zarabia >50K$ rocznie na podstawie cech demograficznych.

**Źródło:** UCI ML Repository (https://archive.ics.uci.edu/ml/datasets/adult)

### Przegląd literatury

#### Praca 1: [Znajdź na Google Scholar]
**Zapytanie:** "Adult Income dataset classification"

**Przykładowe wyniki z literatury:**
- Naive Bayes: ~83%
- Decision Trees: ~85%
- Random Forest: ~86%
- Neural Networks: ~85-87%

### Podsumowanie Adult Income

**Nasze wyniki:**
- Manual MLP: [TODO]
- Keras MLP: [TODO]

**Wnioski:**
- Wyniki powinny być w zakresie 83-87%
- To trudny dataset (silny class imbalance)

---

## DLA STOCK MARKET i STUDENT PERFORMANCE

**Dla regresji:** Trudniej znaleźć bezpośrednie porównania (zbiory są bardziej specyficzne)

**Rozwiązanie:**
1. Opisz dataset
2. Jeśli brak prac - zaznacz to wyraźnie:
   > "Nie znaleziono publikacji wykorzystujących dokładnie ten sam zbiór danych."
3. Przytocz prace o **podobnej tematyce**:
   - Stock market prediction w ogólności
   - Student performance prediction
4. Opisz jakie metody są popularne w tej dziedzinie:
   - Dla stock market: LSTM, GRU, CNN czasowe
   - Dla student performance: Random Forest, XGBoost, Neural Networks

---

## SZABLONY DO WKLEJENIA W RAPORT LaTeX

```latex
\subsection{Adult Income Dataset}

\subsubsection{Przegląd literatury}

Dataset Adult Income jest często wykorzystywany w badaniach nad klasyfikacją binarną.
Poniżej przedstawiono przegląd wybranych prac:

\begin{itemize}
    \item \textbf{[Nazwisko, Rok]} -- Praca wykorzystująca [metoda].
          Osiągnięto accuracy = XX\% \cite{ref1}.
    \item \textbf{[Nazwisko, Rok]} -- Porównanie różnych algorytmów.
          Najlepsze wyniki: [metoda] z accuracy = YY\% \cite{ref2}.
\end{itemize}

\textbf{Podsumowanie wyników z literatury:}
Typowe wyniki dla Adult Income wynoszą 83-87\% accuracy. Najlepsze modele
osiągają około 86-87\%.

\textbf{Nasze wyniki:}
\begin{itemize}
    \item Manual MLP: XX\%
    \item Keras MLP: YY\%
\end{itemize}

Nasze wyniki są [porównywalne/lepsze/gorsze] w stosunku do literatury,
co świadczy o [interpretacja].
```

---

## LITERATURA DO DODANIA W BIBLIOGRAFII

```latex
\bibitem{fashion_mnist}
Xiao, H., Rasul, K., Vollgraf, R. (2017).
\textit{Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms}.
arXiv:1708.07747

\bibitem{adult_income_uci}
Kohavi, R. (1996).
\textit{Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid}.
Proceedings of the Second International Conference on Knowledge Discovery and Data Mining.

% Dodaj więcej po znalezieniu konkretnych prac
```

---

## CO ZROBIĆ TERAZ?

1. **Dla każdego zbioru:**
   - Wyszukaj w Google Scholar
   - Znajdź 2-3 prace
   - Zapisz wyniki (accuracy/MSE) i metody

2. **Wpisz do tego pliku** (jako notatki)

3. **Po zakończeniu eksperymentów:**
   - Porównaj swoje wyniki z literaturą
   - Napisz wnioski

4. **Dodaj do raportu LaTeX**
   - Sekcja 2: Przegląd literatury
   - Użyj szablonów powyżej

---

## ZASOBY

- Google Scholar: https://scholar.google.com
- Papers With Code: https://paperswithcode.com
- Kaggle: https://www.kaggle.com
- UCI ML: https://archive.ics.uci.edu/ml/
- arXiv: https://arxiv.org

**Czas na przegląd literatury: 3-4h**
(~45 min na dataset)
