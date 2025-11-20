"""
MASTER SCRIPT - Uruchamia WSZYSTKIE eksperymenty projektu.

Kolejność:
1. Classification (Adult Income) - Manual MLP + Keras MLP ✅ (już uruchomione wcześniej)
2. Classification Our (Loan Approval) - Manual MLP + Keras MLP ✅ (już uruchomione wcześniej)
3. Regression (Stock Market) - Manual MLP + Keras MLP + CNN 1D + LSTM
4. Regression Our (Student Performance) - Manual MLP + Keras MLP ✅ (już uruchomione wcześniej)
5. Fashion MNIST - Manual MLP + Keras MLP + CNN

UWAGA: Ten skrypt może działać KILKA GODZIN (6-10h w zależności od sprzętu).
Możesz uruchomić każdy moduł osobno jeśli chcesz.
"""

import subprocess
import sys
import time
from datetime import datetime


def run_script(script_name: str, description: str):
    """
    Uruchamia skrypt Pythona i mierzy czas wykonania.

    Args:
        script_name: Nazwa pliku .py do uruchomienia
        description: Opis co robi ten skrypt
    """
    print("\n" + "=" * 80)
    print(f"🚀 URUCHAMIAM: {description}")
    print(f"📝 Skrypt: {script_name}")
    print(f"🕐 Start: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 80 + "\n")

    start_time = time.time()

    try:
        # Uruchom skrypt
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,  # Pokaż output w czasie rzeczywistym
            text=True,
        )

        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        print("\n" + "=" * 80)
        print(f"✅ ZAKOŃCZONO: {description}")
        print(f"⏱️  Czas: {hours}h {minutes}m {seconds}s")
        print(f"🕐 Koniec: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80 + "\n")

        return True, elapsed

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time

        print("\n" + "=" * 80)
        print(f"❌ BŁĄD: {description}")
        print(f"⏱️  Czas do błędu: {elapsed/60:.1f} minut")
        print(f"🔴 Kod błędu: {e.returncode}")
        print("=" * 80 + "\n")

        return False, elapsed

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("⚠️  PRZERWANO przez użytkownika (Ctrl+C)")
        print("=" * 80 + "\n")
        sys.exit(1)


def main():
    print(
        """
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              NEURAL NETWORKS PROJECT - ALL EXPERIMENTS                     ║
║                                                                            ║
║  Ten skrypt uruchomi WSZYSTKIE eksperymenty wymagane w projekcie.        ║
║  Szacowany czas: 6-10 godzin (w zależności od sprzętu)                   ║
║                                                                            ║
║  Możesz uruchomić każdy moduł osobno:                                     ║
║    - python main.py                   (Manual MLP - klasyfikacja/regresja)║
║    - python main_keras.py             (Keras MLP - klasyfikacja/regresja) ║
║    - python main_fashion_mnist.py     (Fashion MNIST - 3 modele)          ║
║    - python main_regression_advanced.py (CNN 1D + LSTM dla regresji)      ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """
    )

    input("\n⏸️  Naciśnij ENTER aby rozpocząć, lub Ctrl+C aby anulować... ")

    experiments = [
        # Moduł 1: Manual MLP (classification + regression)
        {
            "script": "main.py",
            "description": "Manual MLP - Classification & Regression (4 datasets)",
            "note": "Jeśli już uruchomiłeś wcześniej, możesz pominąć (zakomentuj poniżej)",
        },
        # Moduł 2: Keras MLP (classification + regression)
        {
            "script": "main_keras.py",
            "description": "Keras MLP - Classification & Regression (4 datasets)",
            "note": "Z różnymi optymalizatorami (SGD, Adam, RMSprop)",
        },
        # Moduł 3: Fashion MNIST (Manual MLP + Keras MLP + CNN)
        {
            "script": "main_fashion_mnist.py",
            "description": "Fashion MNIST - Manual MLP, Keras MLP, Keras CNN",
            "note": "Analiza obrazów - 3 różne architektury",
        },
        # Moduł 4: Advanced Regression (CNN 1D + LSTM)
        {
            "script": "main_regression_advanced.py",
            "description": "Advanced Regression - CNN 1D & LSTM (Stock Market)",
            "note": "Sieci dla szeregów czasowych",
        },
    ]

    results = []
    total_start = time.time()

    for i, exp in enumerate(experiments, 1):
        print(f"\n📊 MODUŁ {i}/{len(experiments)}")
        print(f"ℹ️  {exp['note']}")

        success, elapsed = run_script(exp["script"], exp["description"])

        results.append(
            {
                "module": i,
                "script": exp["script"],
                "description": exp["description"],
                "success": success,
                "time": elapsed,
            }
        )

        # Podsumowanie po każdym module
        print("\n📈 Postęp:")
        for j, r in enumerate(results, 1):
            status = "✅" if r["success"] else "❌"
            print(f"  {status} Moduł {j}: {r['description']} ({r['time']/60:.1f} min)")

    # KOŃCOWE PODSUMOWANIE
    total_elapsed = time.time() - total_start
    total_hours = int(total_elapsed // 3600)
    total_minutes = int((total_elapsed % 3600) // 60)

    print("\n" + "=" * 80)
    print("🎉 WSZYSTKIE EKSPERYMENTY ZAKOŃCZONE!")
    print("=" * 80)

    print(f"\n⏱️  Całkowity czas: {total_hours}h {total_minutes}m")

    print("\n📊 Podsumowanie:")
    for r in results:
        status = "✅ SUKCES" if r["success"] else "❌ BŁĄD"
        print(f"  {status:12s} | {r['script']:30s} | {r['time']/60:6.1f} min")

    # Sprawdź czy wszystko się powiodło
    all_success = all(r["success"] for r in results)

    if all_success:
        print("\n✅ Wszystkie eksperymenty zakończone pomyślnie!")
    else:
        failed = [r for r in results if not r["success"]]
        print(f"\n⚠️  UWAGA: {len(failed)} moduł(y) zakończyły się błędem:")
        for r in failed:
            print(f"   ❌ {r['script']}")

    print("\n📁 Pliki z wynikami (Excel) powinny być w folderze: results/")
    print("\nKolejne kroki:")
    print("  1. Sprawdź pliki Excel w results/")
    print("  2. Uruchom: python generate_visualizations.py")
    print("  3. Uzupełnij raport: report/raport.tex")
    print("  4. Skompiluj PDF: cd report && make")
    print("\n")


if __name__ == "__main__":
    main()
