from collections.abc import Sequence
import json
from pathlib import Path

import numpy as np

from classificador_de_imagens.config import FIGURES_DIR
from classificador_de_imagens.dataset import CLASS_NAMES


def plot_training_history(
    history: dict[str, Sequence[float]] | Path,
    output_path: Path = FIGURES_DIR / "training_history.png",
) -> Path:
    """Save accuracy and loss curves from a training run."""
    import matplotlib.pyplot as plt

    if isinstance(history, Path):
        history = json.loads(history.read_text(encoding="utf-8"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for key, label in (("accuracy", "Treinamento"), ("val_accuracy", "Validação")):
        if key in history:
            axes[0].plot(history[key], label=label)
    axes[0].set(title="Acurácia por época", xlabel="Época", ylabel="Acurácia")
    axes[0].legend()

    for key, label in (("loss", "Treinamento"), ("val_loss", "Validação")):
        if key in history:
            axes[1].plot(history[key], label=label)
    axes[1].set(title="Perda por época", xlabel="Época", ylabel="Loss")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_predictions(
    images: np.ndarray,
    labels: np.ndarray,
    model,
    output_path: Path = FIGURES_DIR / "predictions.png",
    count: int = 9,
) -> Path:
    """Save a grid comparing true and predicted labels for a batch of images."""
    import matplotlib.pyplot as plt

    if count < 1 or len(images) == 0:
        raise ValueError("At least one image and a positive count are required.")

    count = min(count, len(images))
    probabilities = model.predict(images[:count], verbose=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 3, figsize=(10, 10))
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index >= count:
            continue
        predicted = int(np.argmax(probabilities[index]))
        actual = int(labels[index])
        axis.imshow(images[index])
        axis.set_title(
            f"Real: {CLASS_NAMES[actual]}\n"
            f"Previsto: {CLASS_NAMES[predicted]}\n"
            f"Probabilidade: {probabilities[index, predicted]:.2f}"
        )

    figure.suptitle("Predições CIFAR-10", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
