import json
from pathlib import Path

from loguru import logger
import typer

from classificador_de_imagens.config import METRICS_PATH, MODEL_PATH
from classificador_de_imagens.dataset import load_cifar10
from classificador_de_imagens.modeling.predict import load_trained_model, predict_image
from classificador_de_imagens.modeling.train import train_model
from classificador_de_imagens.plots import plot_predictions, plot_training_history

app = typer.Typer(help="Treinamento e inferência do classificador CIFAR-10.")


@app.command()
def train(
    epochs: int = typer.Option(10, min=1, help="Número de épocas de treinamento."),
    batch_size: int = typer.Option(64, min=1, help="Tamanho do lote."),
) -> None:
    """Treina a CNN e salva o modelo e o histórico."""
    _, history = train_model(epochs=epochs, batch_size=batch_size)
    plot_training_history(history.history)
    logger.success(f"Modelo salvo em {MODEL_PATH}")


@app.command()
def evaluate() -> None:
    """Avalia o modelo salvo no conjunto de teste CIFAR-10."""
    model = load_trained_model()
    (_, _), (images, labels) = load_cifar10()
    results = model.evaluate(images, labels, verbose=0, return_dict=True)
    metrics = {name: float(value) for name, value in results.items()}
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for name, value in metrics.items():
        typer.echo(f"{name}: {value:.4f}")

    output_path = plot_predictions(images, labels, model)
    logger.success(f"Gráfico de predições salvo em {output_path}")


@app.command()
def predict(image_path: Path) -> None:
    """Classifica uma imagem redimensionada para 32 × 32 pixels."""
    label, probability = predict_image(image_path)
    typer.echo(f"Classe: {label} (probabilidade: {probability:.2%})")


if __name__ == "__main__":
    app()
