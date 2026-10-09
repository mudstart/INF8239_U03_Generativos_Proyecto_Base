from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors

from inf8239_u03_gen.data import normalize_images, validate_images
from inf8239_u03_gen.models import build_autoencoder, build_encoder_decoder

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
PANEL_SIZE = 16
CLASS_NAMES = ["Camiseta", "Pantalón", "Pulóver", "Vestido", "Abrigo",
               "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín"]
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_tracker = tf.keras.metrics.Mean(name="reconstruction")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [self.loss_tracker, self.reconstruction_tracker, self.kl_tracker]

    def train_step(self, data):
        images = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            mean, log_variance = self.encoder(images, training=True)
            epsilon = tf.random.normal(tf.shape(mean))
            latent = mean + tf.exp(0.5 * log_variance) * epsilon
            reconstruction = self.decoder(latent, training=True)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(images, reconstruction), axis=(1, 2)))
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_variance - tf.square(mean) - tf.exp(log_variance), axis=1))
            total_loss = reconstruction_loss + kl_loss
        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.loss_tracker.update_state(total_loss)
        self.reconstruction_tracker.update_state(reconstruction_loss)
        self.kl_tracker.update_state(kl_loss)
        return {metric.name: metric.result() for metric in self.metrics}


def image_panel(path: Path, originals, labels, ae_reconstructed, vae_reconstructed, generated):
    row_names = ["Original", "AE", "VAE", "Muestra"]
    columns = len(originals)
    figure, axes = plt.subplots(4, columns, figsize=(columns, 4.4))
    for column in range(columns):
        for row, collection in enumerate([originals, ae_reconstructed, vae_reconstructed, generated]):
            axes[row, column].imshow(collection[column].squeeze(), cmap="gray")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[0, column].set_title(CLASS_NAMES[labels[column]], fontsize=7)
    for row, name in enumerate(row_names):
        axes[row, 0].set_ylabel(name, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=170)


def nearest_neighbor_panel(path: Path, generated, neighbors, neighbor_labels, distances):
    columns = len(generated)
    figure, axes = plt.subplots(2, columns, figsize=(columns, 2.6))
    for column in range(columns):
        axes[0, column].imshow(generated[column].squeeze(), cmap="gray")
        axes[1, column].imshow(neighbors[column].squeeze(), cmap="gray")
        axes[1, column].set_title(f"{CLASS_NAMES[neighbor_labels[column]]}\nd={distances[column]:.2f}", fontsize=6)
        for row in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    axes[0, 0].set_ylabel("Muestra", fontsize=8)
    axes[1, 0].set_ylabel("Vecino", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=170)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-ae", type=int, default=6)
    parser.add_argument("--epochs-vae", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    reports = ROOT / "reports"
    models = ROOT / "models"
    reports.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)
    if args.limit:
        x_train, y_train = x_train[: args.limit], y_train[: args.limit]
        x_test, y_test = x_test[: max(100, args.limit // 5)], y_test[: max(100, args.limit // 5)]
    validate_images(x_train)
    start = perf_counter()
    autoencoder = build_autoencoder(tf)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    autoencoder.fit(x_train, x_train, validation_split=0.1, epochs=args.epochs_ae, batch_size=256,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)], verbose=2)
    ae_seconds = perf_counter() - start
    encoder, decoder = build_encoder_decoder(tf, latent_dim=2)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer="adam")
    start = perf_counter()
    vae.fit(x_train, epochs=args.epochs_vae, batch_size=256, verbose=2)
    vae_seconds = perf_counter() - start
    panel_indices = np.sort(np.random.default_rng(SEED).choice(len(x_test), PANEL_SIZE, replace=False))
    originals = x_test[panel_indices]
    ae_reconstructed = autoencoder.predict(originals, verbose=0)
    mean, log_variance = encoder.predict(originals, verbose=0)
    vae_reconstructed = decoder.predict(mean, verbose=0)
    generated = decoder.predict(np.random.default_rng(SEED).normal(size=(PANEL_SIZE, 2)), verbose=0)
    image_panel(reports / "ae_vae_panel.png", originals, y_test[panel_indices],
                ae_reconstructed, vae_reconstructed, generated)
    search = NearestNeighbors(n_neighbors=1).fit(x_train.reshape(len(x_train), -1))
    distances, neighbor_indices = search.kneighbors(generated.reshape(PANEL_SIZE, -1))
    distances, neighbor_indices = distances[:, 0], neighbor_indices[:, 0]
    nearest_neighbor_panel(reports / "nearest_neighbors.png", generated, x_train[neighbor_indices],
                           y_train[neighbor_indices], distances)
    grid = np.array([(1 - value) * np.array([-2.0, -1.0]) + value * np.array([2.0, 1.0]) for value in np.linspace(0, 1, 12)])
    interpolation = decoder.predict(grid, verbose=0)
    figure, axes = plt.subplots(1, 12, figsize=(12, 1.3))
    for axis, image in zip(axes, interpolation):
        axis.imshow(image.squeeze(), cmap="gray")
        axis.axis("off")
    figure.savefig(reports / "interpolation.png", dpi=170, bbox_inches="tight")
    autoencoder.save(models / "autoencoder.keras")
    decoder.save(models / "decoder.keras")
    metrics = {"ae_train_seconds": ae_seconds, "vae_train_seconds": vae_seconds,
               "ae_parameters": autoencoder.count_params(), "decoder_parameters": decoder.count_params(),
               "panel_test_indices": panel_indices.tolist(),
               "panel_test_classes": [CLASS_NAMES[label] for label in y_test[panel_indices]],
               "nearest_neighbor_train_indices": neighbor_indices.tolist(),
               "nearest_neighbor_classes": [CLASS_NAMES[label] for label in y_train[neighbor_indices]],
               "nearest_neighbor_distances": distances.round(4).tolist()}
    (reports / "training_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
