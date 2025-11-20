from experimetns.advanced_regression_experiments import  run_advanced_regression_experiments
from experimetns.fashion_mnist_experiment import run_fashion_mnist_experiments
from experimetns.run_tabular_keras_experiments import run_tabular_keras_experiments
from experimetns.tabular_manual_mlp_experiments import run_manual_mlp_experiments

if __name__ == "__main__":

    run_manual_mlp_experiments()
    run_tabular_keras_experiments()
    run_advanced_regression_experiments()
    run_fashion_mnist_experiments()
