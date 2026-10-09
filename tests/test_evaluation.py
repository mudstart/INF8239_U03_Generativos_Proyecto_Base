import numpy as np
import pytest

from inf8239_u03_gen.evaluation import (
    diversity_ratio,
    kl_per_dimension,
    per_pixel_metrics,
)


def test_perfect_reconstruction_has_zero_error():
    images = np.array([np.zeros((28, 28, 1)), np.ones((28, 28, 1))], dtype="float32")
    result = per_pixel_metrics(images, images)
    assert result["mse"] == 0
    assert result["bce"] < 1e-5


def test_kl_is_zero_for_reference_distribution():
    result = kl_per_dimension(np.zeros((10, 2)), np.zeros((10, 2)))
    assert result.shape == (2,)
    assert np.allclose(result, 0)


def test_identical_samples_have_zero_diversity():
    rng = np.random.default_rng(0)
    real = rng.random((20, 28, 28, 1))
    generated = np.repeat(real[:1], 20, axis=0)
    assert diversity_ratio(generated, real)["ratio"] == pytest.approx(0, abs=1e-6)
    assert diversity_ratio(real, real)["ratio"] == pytest.approx(1)
