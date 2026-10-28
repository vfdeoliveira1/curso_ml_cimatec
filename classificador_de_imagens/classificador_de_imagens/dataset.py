import numpy as np

CLASS_NAMES = (
    "avião",
    "automóvel",
    "pássaro",
    "gato",
    "cervo",
    "cachorro",
    "sapo",
    "cavalo",
    "navio",
    "caminhão",
)
IMAGE_SIZE = (32, 32)
NUM_CLASSES = len(CLASS_NAMES)


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Convert image pixels to float32 values in the [0, 1] range."""
    if images.ndim != 4 or images.shape[1:] != (*IMAGE_SIZE, 3):
        raise ValueError("Images must have shape (n, 32, 32, 3).")

    normalized = images.astype(np.float32)
    if normalized.size and (normalized.min() < 0 or normalized.max() > 255):
        raise ValueError("Image pixel values must be between 0 and 255.")
    if normalized.size and normalized.max() > 1:
        normalized /= 255.0
    return normalized


def flatten_labels(labels: np.ndarray) -> np.ndarray:
    """Convert CIFAR-10 labels from column vectors to one-dimensional arrays."""
    return np.asarray(labels).reshape(-1).astype(np.int64)


def load_cifar10() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Download (if needed) and return normalized CIFAR-10 train and test data."""
    from tensorflow.keras.datasets import cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return (
        (normalize_images(x_train), flatten_labels(y_train)),
        (normalize_images(x_test), flatten_labels(y_test)),
    )
