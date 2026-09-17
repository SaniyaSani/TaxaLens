from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from scripts.add_local_specimens import infer_view_type
from scripts.embed_multiview import crops_for, fuse_specimens
from diptera_id.hierarchy import load_classifier_bundle


ROOT = Path(__file__).resolve().parents[1]


def test_view_inference_and_multicrop_geometry():
    assert infer_view_type("voucher-1_wing.jpg") == "wing"
    assert infer_view_type("voucher-1_terminalia.png") == "terminalia"
    assert infer_view_type("voucher-1.jpg") == "habitus"
    image = Image.new("RGB", (101, 77), "white")
    crops = crops_for(image, tile_grid=2, include_whole=True)
    assert len(crops) == 5
    assert crops[0].size == (101, 77)


def test_multiview_fusion_keeps_one_vector_per_specimen():
    frame = pd.DataFrame([
        {"specimen_group_id": "a", "view_type": "dorsal", "family": "Phoridae"},
        {"specimen_group_id": "a", "view_type": "wing", "family": "Phoridae"},
        {"specimen_group_id": "b", "view_type": "habitus", "family": "Sciaridae"},
    ]).fillna("")
    vectors = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    fused_frame, fused_vectors = fuse_specimens(frame, vectors)
    assert len(fused_frame) == 2
    assert fused_vectors.shape == (2, 2)
    specimen_a = fused_frame[fused_frame["specimen_group_id"] == "a"].iloc[0]
    assert specimen_a["view_count"] == 2
    assert specimen_a["available_views"] == "dorsal,wing"


def test_blank_specimen_ids_never_collapse_unrelated_records():
    frame = pd.DataFrame([
        {"specimen_group_id": "", "record_id": "one", "view_type": "habitus", "family": "Muscidae"},
        {"specimen_group_id": "", "record_id": "two", "view_type": "habitus", "family": "Muscidae"},
    ])
    fused_frame, fused_vectors = fuse_specimens(frame, np.eye(2, dtype=np.float32))
    assert len(fused_frame) == len(fused_vectors) == 2
    assert fused_frame["specimen_group_id"].is_unique
    assert sorted(fused_frame["embedding_view_ids"]) == ["one", "two"]


def test_hierarchical_training_and_inference(tmp_path: Path):
    rows = []
    vectors = []
    families = ["Phoridae", "Sciaridae"]
    genera = {"Phoridae": ["Megaselia", "Phora"], "Sciaridae": ["Bradysia", "Sciara"]}
    basis = {
        "Megaselia": np.array([1, 0, 1, 0, 1, 0], dtype=np.float32),
        "Phora": np.array([1, 0, 0, 1, 0, 1], dtype=np.float32),
        "Bradysia": np.array([0, 1, 1, 0, 0, 1], dtype=np.float32),
        "Sciara": np.array([0, 1, 0, 1, 1, 0], dtype=np.float32),
    }
    for family in families:
        for genus in genera[family]:
            for species_index in range(2):
                species = f"{genus} species{species_index}"
                for specimen_index in range(6):
                    split = "train" if specimen_index < 4 else ("val" if specimen_index == 4 else "test")
                    rows.append({
                        "family": family,
                        "genus": genus,
                        "species": species,
                        "source": "BIOSCAN-5M" if specimen_index % 2 else "local_verified",
                        "label_quality": "A",
                        "split": split,
                        "split_group": f"{family}-{genus}-{species_index}-{specimen_index}",
                    })
                    vector = basis[genus].copy()
                    vector += np.array([0, 0, species_index * 0.05, 0, 0, specimen_index * 0.001], dtype=np.float32)
                    vector /= np.linalg.norm(vector)
                    vectors.append(vector)
    pd.DataFrame(rows).to_csv(tmp_path / "embedded_manifest.csv", index=False)
    np.save(tmp_path / "embeddings.npy", np.stack(vectors))
    subprocess.run([
        sys.executable,
        str(ROOT / "scripts/train_hierarchical.py"),
        "--model-dir", str(tmp_path),
        "--min-family", "2",
        "--min-genus", "2",
        "--min-species", "2",
    ], cwd=ROOT, check=True)
    bundle = load_classifier_bundle(tmp_path / "classifiers.joblib")
    prediction = bundle.predict_all(np.asarray(vectors[0]), top_k=2)
    assert prediction["family"][0]["taxon"] == "Phoridae"
    assert prediction["genus"][0]["taxon"] == "Megaselia"
    assert prediction["species"]
