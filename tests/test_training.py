import json
import sys
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_ae_vae

from inf8239_u03_gen.evaluation import mean_ssim

ARTIFACTS = ["ae_vae_panel.png", "interpolation.png", "nearest_neighbors.png", "loss_curves.png",
             "per_class_mse.png", "latent_space.png", "results_table.md", "training_metrics.json"]
QUICK_RUN = ["--limit", "600", "--epochs-ae", "1", "--epochs-vae", "1"]


def run_into(root: Path, monkeypatch) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(train_ae_vae, "ROOT", root)
    train_ae_vae.main(QUICK_RUN)
    return json.loads((root / "reports" / "training_metrics.json").read_text(encoding="utf-8"))


def test_ssim_is_one_for_identical_images():
    images = np.random.default_rng(0).random((4, 28, 28, 1)).astype("float32")
    assert mean_ssim(tf, images, images) == pytest.approx(1.0, abs=1e-5)


def test_training_generates_every_artifact(tmp_path, monkeypatch):
    metrics = run_into(tmp_path, monkeypatch)
    for name in ARTIFACTS:
        assert (tmp_path / "reports" / name).is_file(), name
    for name in ["autoencoder.keras", "decoder.keras"]:
        assert (tmp_path / "models" / name).is_file(), name
    assert len(metrics["panel_test_indices"]) == train_ae_vae.PANEL_SIZE
    assert 0 < metrics["test_fidelity"]["vae"]["ssim"] <= 1


def test_training_is_reproducible_across_runs(tmp_path, monkeypatch):
    first = run_into(tmp_path / "first", monkeypatch)
    second = run_into(tmp_path / "second", monkeypatch)
    for key in ["loss_history", "test_fidelity", "nearest_neighbor_distances", "diversity"]:
        assert first[key] == second[key], key
