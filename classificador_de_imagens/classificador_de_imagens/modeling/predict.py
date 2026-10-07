from pathlib import Path

import numpy as np

from classificador_de_imagens.config import MODEL_PATH
from classificador_de_imagens.dataset import CLASS_NAMES, load_cifar10
from classificador_de_imagens.features import load_image


def load_trained_model(model_path: Path = MODEL_PATH):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run `python -m classificador_de_imagens train` first."
        )

    from tensorflow import keras

    return keras.models.load_model(model_path)


def predict_image(image_path: Path, model_path: Path = MODEL_PATH) -> tuple[str, float]:
    """Return the predicted CIFAR-10 class and its probability for an image."""
    image = load_image(image_path)
    model = load_trained_model(model_path)
    probabilities = model.predict(image, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    return CLASS_NAMES[predicted_index], float(probabilities[predicted_index])


def evaluate_model(model_path: Path = MODEL_PATH) -> dict[str, float]:
    """Evaluate the saved model against CIFAR-10's held-out test split."""
    model = load_trained_model(model_path)
    (_, _), (x_test, y_test) = load_cifar10()
    results = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    return {name: float(value) for name, value in results.items()}
