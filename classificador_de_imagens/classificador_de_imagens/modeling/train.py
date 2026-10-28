import json
from pathlib import Path

from classificador_de_imagens.config import HISTORY_PATH, MODEL_PATH
from classificador_de_imagens.dataset import load_cifar10


def build_model():
    """Build the convolutional network used by the CIFAR-10 notebook."""
    from tensorflow import keras
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D

    model = keras.Sequential(
        [
            Input(shape=(32, 32, 3)),
            Conv2D(filters=16, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=64, activation="relu"),
            Dense(units=10, activation="softmax"),
        ],
        name="cifar10_classifier",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    epochs: int = 10,
    batch_size: int = 64,
    model_path: Path = MODEL_PATH,
    history_path: Path = HISTORY_PATH,
):
    """Train the CNN, save it in Keras format, and persist its learning curves."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    (x_train, y_train), _ = load_cifar10()
    model = build_model()
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    history_path.write_text(
        json.dumps(history.history, indent=2),
        encoding="utf-8",
    )
    return model, history
