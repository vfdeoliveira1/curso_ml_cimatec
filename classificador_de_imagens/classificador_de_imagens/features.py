from pathlib import Path

import numpy as np

from classificador_de_imagens.dataset import IMAGE_SIZE, normalize_images


def load_image(image_path: Path) -> np.ndarray:
    """Load an image, resize it to CIFAR-10 dimensions, and prepare a model batch."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    from tensorflow.keras.utils import img_to_array, load_img

    image = load_img(image_path, target_size=IMAGE_SIZE, color_mode="rgb")
    image_array = img_to_array(image)
    return normalize_images(np.expand_dims(image_array, axis=0))
