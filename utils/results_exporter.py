from datetime import datetime

import pandas as pd


class ResultsExporter:
    def __init__(self, filename="results_manual_mlp.xlsx"):
        import os

        # Jeśli filename nie zawiera ścieżki, dodaj 'results/' jako prefix
        if not os.path.dirname(filename):
            filename = os.path.join("results", filename)
        self.filename = filename

    def export(
        self, results_dict, params_dict=None, description=None, training_time=None
    ):
        """
        Exports all metrics (e.g. loss, accuracy, precision, recall, etc.) and model parameters to Excel file.
        - results_dict: dict with metrics (e.g. {"epoch": [...], "loss": [...], ...})
        - params_dict: dict with model parameters (e.g. {"layers": 2, ...})
        - description: optional experiment description
        - training_time: training duration (seconds)
        """
        with pd.ExcelWriter(self.filename) as writer:
            # Wyniki: tylko hiperparametry, run i metryki
            df_results = pd.DataFrame(results_dict)
            df_results.to_excel(writer, sheet_name="Results", index=False)

            # Parametry eksperymentu: tylko description, date, training_time
            params_info = {}
            if description:
                params_info["description"] = description
            params_info["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if training_time is not None:
                params_info["training_time_sec"] = training_time
            df_params = pd.DataFrame([params_info])
            df_params.to_excel(writer, sheet_name="Params", index=False)

        print(f"Results and parameters saved to {self.filename}")
