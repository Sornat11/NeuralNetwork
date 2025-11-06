from datetime import datetime

import pandas as pd


class ResultsExporter:
    def __init__(self, filename="results.xlsx"):
        self.filename = filename

    def export(
        self, results_dict, params_dict=None, description=None, training_time=None
    ):
        """
        Eksportuje wyniki (np. loss, accuracy) oraz parametry modelu do pliku Excel.
        - results_dict: słownik z metrykami (np. {"epoch": [...], "loss": [...], ...})
        - params_dict: słownik z parametrami modelu (np. {"layers": 2, ...})
        - description: opcjonalny opis eksperymentu
        - training_time: czas trwania treningu (sekundy)
        """
        with pd.ExcelWriter(self.filename) as writer:
            # Parametry, opis i czas treningu jako nagłówek w arkuszu Results
            header = {}
            if params_dict:
                header.update(params_dict)
            if description:
                header["description"] = description
            header["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if training_time is not None:
                header["training_time_sec"] = training_time
            # Zamiana na DataFrame z jednego wiersza
            df_header = pd.DataFrame([header])
            df_results = pd.DataFrame(results_dict)
            # Zapisz nagłówek w wierszu 1, wyniki od wiersza 2
            df_header.to_excel(writer, sheet_name="Results", index=False, startrow=0)
            df_results.to_excel(writer, sheet_name="Results", index=False, startrow=1)

            # Parametry modelu i eksperymentu w osobnym arkuszu
            if params_dict is not None:
                df_params = pd.DataFrame([params_dict])
                df_params["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if description:
                    df_params["description"] = description
                if training_time is not None:
                    df_params["training_time_sec"] = training_time
                df_params.to_excel(writer, sheet_name="Params", index=False)

        print(f"Wyniki i parametry zapisane do {self.filename}")
