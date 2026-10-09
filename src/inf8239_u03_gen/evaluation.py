from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances


def per_pixel_metrics(originals: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    clipped = np.clip(reconstructed, 1e-7, 1 - 1e-7)
    bce = -np.mean(originals * np.log(clipped) + (1 - originals) * np.log(1 - clipped))
    mse = np.mean((originals - reconstructed) ** 2)
    return {"bce": float(bce), "mse": float(mse)}


def kl_per_dimension(mean: np.ndarray, log_variance: np.ndarray) -> np.ndarray:
    return -0.5 * np.mean(1 + log_variance - mean**2 - np.exp(log_variance), axis=0)


def mean_pairwise_distance(images: np.ndarray) -> float:
    distances = pairwise_distances(images.reshape(len(images), -1))
    return float(distances[np.triu_indices(len(images), 1)].mean())


def diversity_ratio(generated: np.ndarray, real: np.ndarray) -> dict[str, float]:
    generated_distance = mean_pairwise_distance(generated)
    real_distance = mean_pairwise_distance(real)
    return {"generated_mean_pairwise": generated_distance, "real_mean_pairwise": real_distance,
            "ratio": generated_distance / real_distance}
