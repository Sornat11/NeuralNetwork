

# Neural Network Playground: Classification & Regression

## Overview
This project is a modular, educational framework for experimenting with neural networks in Python. It supports both classic (NumPy-based) and modern (Keras-based) models for classification and regression tasks, with a clean CLI and comprehensive test suite.

---


## Features

- Manual implementation of neural networks (MLP, modular layers, activations)
- Batch training, L1/L2 regularization, metrics logging
- Automated hyperparameter experiments (ExperimentRunner)
- Results export to Excel
- Ready for plugging in new models (MLP, CNN, RNN, etc.)
- CLI-ready structure
- Unit tests for key components

---

## Getting Started


### Requirements


#### Create a Virtual Environment (Recommended)

On Windows:
```bash
python -m venv venv
 .\.venv\Scripts\Activate.ps1
```
On Linux/Mac:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```



### Usage Example

Run the main experiment script:
```bash
python main.py
```

To add and compare new models, pass their class to ExperimentRunner in `main.py`.

Progress of experiments is shown via tqdm progress bar.

Results are exported to `experiment_results.xlsx`.

See also `ToDo.md` for planned improvements.


## Project Structure

```
main.py                # Main experiment runner
requirements.txt       # Dependencies
src/manual_mlp/        # Manual neural network implementation
src/models/            # Other model types (CNN, RNN, etc.)
utils/experiment_runner.py  # Experiment automation
utils/results_exporter.py   # Excel export
data/                  # Sample data generators and datasets
tests/                 # Unit tests
notebooks/             # Jupyter notebooks
results/               # Output results
ToDo.md                # Planned tasks
```


## Authors

- **Jakub Sornat**
- **Maciej Tajs**
- **Bartłomiej Sadza**

---

## License
This project is for educational purposes only.

## Project Report

[Project report (DOCX)](https://aghedupl-my.sharepoint.com/:w:/r/personal/jakubsornat_student_agh_edu_pl/Documents/report.docx?d=w719a3c159b694350a6cdfea27e91fec0&csf=1&web=1&e=UyxR3n)