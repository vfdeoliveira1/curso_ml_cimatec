import numpy as np
import pytest

from classificador_de_imagens.dataset import (
    CLASS_NAMES,
    flatten_labels,
    normalize_images,
)


def test_normalize_images_scales_cifar_pixels_to_zero_one() -> None:
    images = np.full((2, 32, 32, 3), 255, dtype=np.uint8)

    normalized = normalize_images(images)

    assert normalized.shape == images.shape
    assert normalized.dtype == np.float32
    assert np.all(normalized == 1.0)


def test_normalize_images_rejects_non_cifar_shapes() -> None:
    with pytest.raises(ValueError, match="32, 32, 3"):
        normalize_images(np.zeros((1, 28, 28, 3), dtype=np.uint8))


def test_normalize_images_rejects_pixels_outside_valid_range() -> None:
    images = np.full((1, 32, 32, 3), 256, dtype=np.uint16)

    with pytest.raises(ValueError, match="between 0 and 255"):
        normalize_images(images)


def test_flatten_labels_converts_column_to_integer_vector() -> None:
    labels = flatten_labels(np.array([[2], [9]], dtype=np.uint8))

    np.testing.assert_array_equal(labels, [2, 9])
    assert labels.dtype == np.int64


def test_cifar10_has_ten_labels() -> None:
    assert len(CLASS_NAMES) == 10
    assert CLASS_NAMES[0] == "avião"
