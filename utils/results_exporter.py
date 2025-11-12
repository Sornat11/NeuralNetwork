from datetime import datetime

import pandas as pd


class ResultsExporter:
    def __init__(self, filename="results_manual_mlp.xlsx"):
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
            # Header: parameters, description, training time in Results sheet
            header = {}
            if params_dict:
                header.update(params_dict)
            if description:
                header["description"] = description
            header["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if training_time is not None:
                header["training_time_sec"] = training_time
            df_header = pd.DataFrame([header])
            df_results = pd.DataFrame(results_dict)
            # Write header in row 1, results from row 2
            df_header.to_excel(writer, sheet_name="Results", index=False, startrow=0)
            df_results.to_excel(writer, sheet_name="Results", index=False, startrow=1)

            # Model parameters and experiment info in separate sheet
            if params_dict is not None:
                df_params = pd.DataFrame([params_dict])
                df_params["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if description:
                    df_params["description"] = description
                if training_time is not None:
                    df_params["training_time_sec"] = training_time
                df_params.to_excel(writer, sheet_name="Params", index=False)

        print(f"Results and parameters saved to {self.filename}")
