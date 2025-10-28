from data.sample_data_generator import create_data
from src.manual_mlp.activations import ActivationReLU, ActivationSoftmax
from src.manual_mlp.layers import LayerDense
from src.manual_mlp.losses import LossCategoricalCrossentropy
from src.manual_mlp.model import Model
from utils.seed import set_seed


def main():
    set_seed(0)

    # dane
    X, y = create_data(100, 3)

    # definicja modelu
    model = Model()
    model.add(LayerDense(2, 3))
    model.add(ActivationReLU())
    model.add(LayerDense(3, 3))
    model.add(ActivationSoftmax())

    # forward pass
    y_pred = model.forward(X)

    # strata
    loss_fn = LossCategoricalCrossentropy()
    loss = loss_fn.calculate(y_pred, y)
    print("Próbka predykcji:\n", y_pred[:5])
    print("Loss:", loss)


if __name__ == "__main__":
    main()
