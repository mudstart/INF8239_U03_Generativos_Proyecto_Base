from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors

from inf8239_u03_gen.data import normalize_images, validate_images
from inf8239_u03_gen.evaluation import (
    diversity_ratio,
    kl_per_dimension,
    per_pixel_metrics,
)
from inf8239_u03_gen.models import build_autoencoder, build_encoder_decoder

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
PANEL_SIZE = 16
DIVERSITY_SAMPLES = 500
CLASS_NAMES = ["Camiseta", "Pantalón", "Pulóver", "Vestido", "Abrigo",
               "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín"]


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
    plt.close(figure)


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
    plt.close(figure)


def loss_curves(path: Path, history_ae, history_vae):
    figure, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    epochs_ae = range(1, len(history_ae["loss"]) + 1)
    axes[0].plot(epochs_ae, history_ae["loss"], marker="o", label="entrenamiento")
    axes[0].plot(epochs_ae, history_ae["val_loss"], marker="o", label="validación")
    axes[0].set(title="AE · BCE por píxel", xlabel="época")
    axes[0].legend()
    epochs_vae = range(1, len(history_vae["loss"]) + 1)
    axes[1].plot(epochs_vae, history_vae["reconstruction"], marker="o", color="tab:green")
    axes[1].set(title="VAE · reconstrucción (BCE sumada)", xlabel="época")
    axes[2].plot(epochs_vae, history_vae["kl"], marker="o", color="tab:red")
    axes[2].set(title="VAE · KL", xlabel="época")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def per_class_panel(path: Path, per_class_mse):
    positions = np.arange(len(CLASS_NAMES))
    figure, axis = plt.subplots(figsize=(11, 3.5))
    axis.bar(positions - 0.2, list(per_class_mse["autoencoder"].values()), width=0.4, label="AE")
    axis.bar(positions + 0.2, list(per_class_mse["vae"].values()), width=0.4, label="VAE")
    axis.set_xticks(positions, CLASS_NAMES, rotation=30)
    axis.set(title="MSE de reconstrucción por clase (prueba)", ylabel="MSE/píxel")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def latent_space_panel(path: Path, mean, labels):
    figure, axis = plt.subplots(figsize=(7, 6))
    scatter = axis.scatter(mean[:, 0], mean[:, 1], c=labels, cmap="tab10", vmin=0, vmax=9, s=2, alpha=0.6)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=scatter.cmap(scatter.norm(index)), label=name)
               for index, name in enumerate(CLASS_NAMES)]
    axis.legend(handles=handles, fontsize=7, loc="upper right", markerscale=1.2)
    axis.set(title="Media del posterior (z_mean) · conjunto de prueba", xlabel="z₁", ylabel="z₂")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def results_table(path: Path, metrics):
    fidelity, diversity = metrics["test_fidelity"], metrics["diversity"]
    rows = [
        ("BCE/píxel · prueba", f"{fidelity['autoencoder']['bce']:.4f}", f"{fidelity['vae']['bce']:.4f}"),
        ("MSE/píxel · prueba", f"{fidelity['autoencoder']['mse']:.4f}", f"{fidelity['vae']['mse']:.4f}"),
        ("Mejor val_loss (BCE/píxel)", f"{metrics['ae_best_val_loss']:.4f}", "— (sin validación)"),
        ("KL final (entrenamiento)", "—", f"{metrics['vae_final_kl']:.2f}"),
        ("Genera desde N(0, I)", "No", "Sí"),
        ("Mediana dist. vecino (muestras / prueba real)", "—",
         f"{np.median(metrics['nearest_neighbor_distances']):.2f} / {np.median(metrics['test_reference_distances']):.2f}"),
        ("Razón de diversidad (generadas / reales)", "—", f"{diversity['ratio']:.2f}"),
        ("Parámetros", f"{metrics['ae_parameters']:,}", f"{metrics['encoder_parameters'] + metrics['decoder_parameters']:,}"),
        ("Tiempo de entrenamiento CPU (s)", f"{metrics['ae_train_seconds']:.1f}", f"{metrics['vae_train_seconds']:.1f}"),
    ]
    lines = ["| Métrica | AE | VAE |", "|---|---|---|"] + [f"| {name} | {ae} | {vae} |" for name, ae, vae in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-ae", type=int, default=6)
    parser.add_argument("--epochs-vae", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
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
    validate_images(x_test)
    start = perf_counter()
    autoencoder = build_autoencoder(tf)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    history_ae = autoencoder.fit(x_train, x_train, validation_split=0.1, epochs=args.epochs_ae, batch_size=256,
                                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)], verbose=2)
    ae_seconds = perf_counter() - start
    encoder, decoder = build_encoder_decoder(tf, latent_dim=2)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer="adam")
    start = perf_counter()
    history_vae = vae.fit(x_train, epochs=args.epochs_vae, batch_size=256, verbose=2)
    vae_seconds = perf_counter() - start
    history_ae, history_vae = history_ae.history, history_vae.history
    loss_curves(reports / "loss_curves.png", history_ae, history_vae)

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
    test_reference_distances = search.kneighbors(originals.reshape(PANEL_SIZE, -1))[0][:, 0]
    grid = np.array([(1 - value) * np.array([-2.0, -1.0]) + value * np.array([2.0, 1.0]) for value in np.linspace(0, 1, 12)])
    interpolation = decoder.predict(grid, verbose=0)
    figure, axes = plt.subplots(1, 12, figsize=(12, 1.3))
    for axis, image in zip(axes, interpolation):
        axis.imshow(image.squeeze(), cmap="gray")
        axis.axis("off")
    figure.savefig(reports / "interpolation.png", dpi=170, bbox_inches="tight")
    plt.close(figure)

    ae_test = autoencoder.predict(x_test, batch_size=1024, verbose=0)
    test_mean, test_log_variance = encoder.predict(x_test, batch_size=1024, verbose=0)
    vae_test = decoder.predict(test_mean, batch_size=1024, verbose=0)
    per_class_mse = {name: {CLASS_NAMES[label]: per_pixel_metrics(x_test[y_test == label], output[y_test == label])["mse"]
                            for label in range(len(CLASS_NAMES))}
                     for name, output in [("autoencoder", ae_test), ("vae", vae_test)]}
    per_class_panel(reports / "per_class_mse.png", per_class_mse)
    latent_space_panel(reports / "latent_space.png", test_mean, y_test)
    rng = np.random.default_rng(SEED)
    diversity_samples = decoder.predict(rng.normal(size=(DIVERSITY_SAMPLES, 2)), verbose=0)
    real_samples = x_test[rng.choice(len(x_test), min(DIVERSITY_SAMPLES, len(x_test)), replace=False)]

    autoencoder.save(models / "autoencoder.keras")
    decoder.save(models / "decoder.keras")
    metrics = {"ae_train_seconds": ae_seconds, "vae_train_seconds": vae_seconds,
               "ae_parameters": autoencoder.count_params(), "encoder_parameters": encoder.count_params(),
               "decoder_parameters": decoder.count_params(),
               "partitions": {"train": len(x_train), "ae_validation_fraction": 0.1, "test": len(x_test)},
               "ae_best_val_loss": min(history_ae["val_loss"]),
               "ae_best_epoch": int(np.argmin(history_ae["val_loss"])) + 1,
               "vae_final_reconstruction": history_vae["reconstruction"][-1],
               "vae_final_kl": history_vae["kl"][-1],
               "loss_history": {"autoencoder": history_ae, "vae": history_vae},
               "test_fidelity": {"autoencoder": per_pixel_metrics(x_test, ae_test),
                                 "vae": per_pixel_metrics(x_test, vae_test)},
               "vae_test_kl_per_dimension": kl_per_dimension(test_mean, test_log_variance).round(4).tolist(),
               "per_class_mse": per_class_mse,
               "panel_test_indices": panel_indices.tolist(),
               "panel_test_classes": [CLASS_NAMES[label] for label in y_test[panel_indices]],
               "nearest_neighbor_train_indices": neighbor_indices.tolist(),
               "nearest_neighbor_classes": [CLASS_NAMES[label] for label in y_train[neighbor_indices]],
               "nearest_neighbor_distances": distances.round(4).tolist(),
               "test_reference_distances": test_reference_distances.round(4).tolist(),
               "diversity": diversity_ratio(diversity_samples, real_samples)}
    metrics = json.loads(json.dumps(metrics, default=float))
    (reports / "training_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    results_table(reports / "results_table.md", metrics)
    summary = {key: metrics[key] for key in ["ae_train_seconds", "vae_train_seconds", "ae_best_val_loss", "test_fidelity", "diversity"]}
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
