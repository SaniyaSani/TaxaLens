from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diptera_id.embedding import FusionEmbedder
from scripts.fuse_embedding_models import fuse
from scripts.embedding_safety import checkpoint_ready, digest, cohort_indices


def write_model(path: Path, ids: list[str], vectors: np.ndarray, backend: str) -> None:
    path.mkdir(parents=True)
    pd.DataFrame({
        "specimen_group_id": ids,
        "record_id": [f"record:{value}" for value in ids],
        "family": ["Muscidae"] * len(ids),
    }).to_csv(path / "embedded_manifest.csv", index=False)
    np.save(path / "embeddings.npy", vectors.astype(np.float32))
    (path / "embedding_config.json").write_text(json.dumps({
        "backend": backend,
        "backbone": f"fake-{backend}",
        "image_size": 224,
        "tile_grid": 1,
        "include_whole": True,
    }))


def test_fusion_aligns_by_specimen_and_uses_intersection(tmp_path: Path):
    dino = tmp_path / "dino"
    bio = tmp_path / "bio"
    out = tmp_path / "fusion"
    write_model(dino, ["a", "b"], np.array([[1, 0], [0, 1]]), "dino")
    write_model(bio, ["b", "c"], np.array([[2, 0, 0], [0, 3, 0]]), "bioclip")
    config = fuse(dino, bio, out)
    frame = pd.read_csv(out / "embedded_manifest.csv")
    vectors = np.load(out / "embeddings.npy")
    assert frame["specimen_group_id"].tolist() == ["b"]
    assert vectors.shape == (1, 5)
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0)
    assert config["dino_only_specimens"] == 1
    assert config["bioclip_only_specimens"] == 1
    assert len(config["cohort_sha256"]) == 64


def test_fusion_rejects_taxonomy_mismatch(tmp_path: Path):
    dino = tmp_path / "dino"
    bio = tmp_path / "bio"
    write_model(dino, ["a"], np.array([[1, 0]]), "dino")
    write_model(bio, ["a"], np.array([[1, 0]]), "bioclip")
    frame = pd.read_csv(bio / "embedded_manifest.csv")
    frame["family"] = "Tachinidae"
    frame.to_csv(bio / "embedded_manifest.csv", index=False)
    with pytest.raises(SystemExit, match="metadata differs"):
        fuse(dino, bio, tmp_path / "fusion")


def test_fusion_rejects_duplicate_preferred_keys(tmp_path: Path):
    dino = tmp_path / "dino"
    bio = tmp_path / "bio"
    write_model(dino, ["a", "a"], np.eye(2), "dino")
    write_model(bio, ["a", "b"], np.eye(2), "bioclip")
    with pytest.raises(SystemExit, match="duplicate specimen keys"):
        fuse(dino, bio, tmp_path / "fusion")


def test_equal_weight_join_normalizes_each_encoder():
    fused = FusionEmbedder._join([
        np.array([10.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 5.0], dtype=np.float32),
    ])
    assert np.allclose(fused, np.array([2 ** -0.5, 0, 0, 0, 2 ** -0.5]))


def test_checkpoint_requires_current_manifest_and_intact_artifacts(tmp_path: Path):
    manifest = tmp_path / "shard.parquet"
    pd.DataFrame({"record_id": ["one"]}).to_parquet(manifest, index=False)
    out = tmp_path / "out"
    out.mkdir()
    np.save(out / "embeddings.npy", np.array([[1.0, 0.0]], dtype=np.float16))
    pd.DataFrame({"record_id": ["one"]}).to_csv(out / "embedded_manifest.csv", index=False)
    config = {"backend": "dino", "model": "fake", "image_size": 512,
              "tile_grid": 1, "include_whole": True, "local_only": True}
    complete = out / "complete.json"
    complete.write_text(json.dumps({
        "manifest_sha256": digest(manifest),
        "embedding": {"backend": "dino", "backbone": "fake", "image_size": 512,
                      "tile_grid": 1, "include_whole": True, "local_only": True,
                      "preprocessing": "square_pad_v1"},
        "artifact_sha256": {
            "embeddings.npy": digest(out / "embeddings.npy"),
            "embedded_manifest.csv": digest(out / "embedded_manifest.csv"),
        },
    }))
    assert checkpoint_ready(complete, manifest, config)
    with (out / "embeddings.npy").open("ab") as handle:
        handle.write(b"interrupted")
    assert not checkpoint_ready(complete, manifest, config)


def test_shared_cohort_rejects_changed_labels():
    frame = pd.DataFrame({"specimen_group_id": ["a"], "family": ["Muscidae"]})
    cohort = pd.DataFrame({"specimen_group_id": ["a"], "family": ["Tachinidae"]})
    with pytest.raises(SystemExit, match="metadata differs"):
        cohort_indices(frame, cohort)
