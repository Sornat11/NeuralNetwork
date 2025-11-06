"""
Główny punkt wejścia do projektu Neural Network.
Prosty demonstrator funkcjonalności.
"""

import numpy as np
from data.datasets import get_dataset
from src.manual_mlp.model import Model
from src.manual_mlp.layers import LayerDense
from src.manual_mlp.activations import ActivationReLU
from src.manual_mlp.losses import SoftmaxCategoricalCrossentropy
from src.manual_mlp.optimizers import OptimizerAdam
from src.manual_mlp.metrics import evaluate_classification
from utils.seed import set_seed


def demo_custom_mlp():
    """Demonstracja własnej implementacji MLP na datasecie Iris"""

    print("\n" + "="*80)
    print("DEMONSTRACJA: Własna implementacja MLP na Iris")
    print("="*80 + "\n")

    set_seed(42)

    # Załaduj dane
    print("Ładowanie datasetu Iris...")
    data = get_dataset("iris", split_type="70_15_15", normalize=True)

    print(f"✓ Train: {data['X_train'].shape}")
    print(f"✓ Val: {data['X_val'].shape}")
    print(f"✓ Test: {data['X_test'].shape}")
    print(f"✓ Klasy: {data['target_names']}\n")

    # Zbuduj model
    print("Budowanie modelu...")
    model = Model()
    model.add(LayerDense(data['n_features'], 64))
    model.add(ActivationReLU())
    model.add(LayerDense(64, 32))
    model.add(ActivationReLU())
    model.add(LayerDense(32, data['n_classes']))

    # Skonfiguruj
    loss = SoftmaxCategoricalCrossentropy()
    optimizer = OptimizerAdam(learning_rate=0.01)

    model.set(loss=loss, optimizer=optimizer)
    model.finalize()

    print("✓ Model zbudowany: 4→64→32→3")
    print("✓ Optimizer: Adam (lr=0.01)")
    print("✓ Loss: Softmax + Categorical Crossentropy\n")

    # Trenuj
    print("Trening modelu (100 epok)...")
    history = model.fit(
        data['X_train'],
        data['y_train'],
        epochs=100,
        batch_size=16,
        validation_data=(data['X_val'], data['y_val']),
        verbose=False
    )

    print(f"✓ Trening zakończony!")
    print(f"  Final train loss: {history['loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}\n")

    # Ewaluacja
    print("Ewaluacja modelu...")

    # Train set
    train_pred = model.predict(data['X_train'])
    train_metrics = evaluate_classification(data['y_train'], train_pred)

    # Val set
    val_pred = model.predict(data['X_val'])
    val_metrics = evaluate_classification(data['y_val'], val_pred)

    # Test set
    test_pred = model.predict(data['X_test'])
    test_metrics = evaluate_classification(data['y_test'], test_pred)

    # Wyniki
    print("\nWYNIKI:")
    print("-" * 80)
    print(f"{'Zbiór':<15} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-" * 80)
    print(f"{'Train':<15} {train_metrics['accuracy']:<15.4f} {train_metrics['precision']:<15.4f} {train_metrics['recall']:<15.4f} {train_metrics['f1_score']:<15.4f}")
    print(f"{'Validation':<15} {val_metrics['accuracy']:<15.4f} {val_metrics['precision']:<15.4f} {val_metrics['recall']:<15.4f} {val_metrics['f1_score']:<15.4f}")
    print(f"{'Test':<15} {test_metrics['accuracy']:<15.4f} {test_metrics['precision']:<15.4f} {test_metrics['recall']:<15.4f} {test_metrics['f1_score']:<15.4f}")
    print("-" * 80)

    print("\n✅ Demo zakończone!")


def print_menu():
    """Wyświetla menu główne"""
    print("\n" + "="*80)
    print("NEURAL NETWORK PROJECT - MENU GŁÓWNE")
    print("="*80)
    print("\nDostępne opcje:\n")
    print("1. Demo: Własny MLP na Iris")
    print("2. Uruchom eksperymenty klasyfikacyjne (Iris + Wine)")
    print("3. Uruchom eksperymenty regresyjne (Airline Passengers)")
    print("4. Uruchom eksperymenty na obrazach (Fashion MNIST)")
    print("5. Informacje o projekcie")
    print("0. Wyjście")
    print("\n" + "="*80)


def print_info():
    """Wyświetla informacje o projekcie"""
    print("\n" + "="*80)
    print("INFORMACJE O PROJEKCIE")
    print("="*80 + "\n")
    print("Projekt: Analiza i porównanie wybranych architektur sieci neuronowych")
    print("Autorzy: Jakub Sornat, Maciej Tajs, Bartłomiej Sadza")
    print("Kierunek: Informatyka i Ekonometria")
    print("Prowadzący: dr inż. Radosław Puka\n")
    print("Opis:")
    print("Projekt porównuje różne architektury sieci neuronowych (MLP, CNN, RNN/LSTM/GRU)")
    print("w trzech typach zadań:")
    print("  1. Klasyfikacja - Iris, Wine")
    print("  2. Regresja - Airline Passengers (szereg czasowy)")
    print("  3. Analiza obrazów - Fashion MNIST\n")
    print("Kluczowe cechy:")
    print("  ✓ Własna implementacja MLP od zera (NumPy)")
    print("  ✓ Optimizery: SGD, Adam, RMSprop z momentum")
    print("  ✓ Framework eksperymentalny z grid search")
    print("  ✓ Wielokrotne powtórzenia (min. 5x)")
    print("  ✓ Metryki: accuracy, precision, recall, F1, MSE, MAE, R²")
    print("  ✓ Wizualizacje: learning curves, confusion matrix\n")
    print("Więcej informacji: README.md")
    print("="*80)


def main():
    """Główna funkcja programu"""

    print("\n" + "="*80)
    print("WITAJ W PROJEKCIE NEURAL NETWORK!")
    print("="*80)
    print("\nProjekt akademicki: Analiza i porównanie architektur sieci neuronowych")
    print("Autorzy: Jakub Sornat, Maciej Tajs, Bartłomiej Sadza\n")

    while True:
        print_menu()

        try:
            choice = input("\nWybierz opcję (0-5): ").strip()

            if choice == "0":
                print("\nDo widzenia! 👋")
                break

            elif choice == "1":
                demo_custom_mlp()

            elif choice == "2":
                print("\n⚠️  Uruchamianie eksperymentów klasyfikacyjnych...")
                print("To może potrwać kilka minut. Wyniki zostaną zapisane w katalogu 'results/'")
                confirm = input("Kontynuować? (t/n): ").strip().lower()
                if confirm == "t":
                    import sys
                    import subprocess
                    subprocess.run([sys.executable, "experiments/run_classification_experiments.py"])
                else:
                    print("Anulowano.")

            elif choice == "3":
                print("\n⚠️  Uruchamianie eksperymentów regresyjnych...")
                print("To może potrwać kilkanaście minut (LSTM/GRU są wolniejsze).")
                confirm = input("Kontynuować? (t/n): ").strip().lower()
                if confirm == "t":
                    import sys
                    import subprocess
                    subprocess.run([sys.executable, "experiments/run_regression_experiments.py"])
                else:
                    print("Anulowano.")

            elif choice == "4":
                print("\n⚠️  Uruchamianie eksperymentów na obrazach...")
                print("To może potrwać 20-30 minut (Fashion MNIST jest duży).")
                confirm = input("Kontynuować? (t/n): ").strip().lower()
                if confirm == "t":
                    import sys
                    import subprocess
                    subprocess.run([sys.executable, "experiments/run_image_experiments.py"])
                else:
                    print("Anulowano.")

            elif choice == "5":
                print_info()

            else:
                print("\n❌ Nieprawidłowa opcja. Wybierz 0-5.")

        except KeyboardInterrupt:
            print("\n\n⚠️  Przerwano przez użytkownika.")
            break
        except Exception as e:
            print(f"\n❌ Błąd: {e}")

    print("\nZamykanie programu...")


if __name__ == "__main__":
    main()
