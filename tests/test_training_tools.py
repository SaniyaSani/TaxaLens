from __future__ import annotations

import json
import gzip
import subprocess
import sys
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from scripts.download_inat_metadata import extract_selected
from scripts.train_multidomain import balanced_weights, top_k_accuracy


def test_inat_archive_extracts_only_expected_files(tmp_path: Path):
    archive = tmp_path / "metadata.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name in ("folder/observations.csv.gz", "photos.csv.gz", "taxa.csv.gz", "observers.csv.gz", "ignore.txt"):
            payload = name.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, BytesIO(payload))
    out = tmp_path / "out"
    paths = extract_selected(archive, out)
    assert {path.name for path in paths} == {"observations.csv.gz", "photos.csv.gz", "taxa.csv.gz", "observers.csv.gz"}
    assert not (out / "ignore.txt").exists()


def test_inat_plain_tar_csv_members_are_normalized_to_gzip(tmp_path: Path):
    archive = tmp_path / "metadata.tar.gz"
    expected = {}
    with tarfile.open(archive, "w:") as bundle:
        for name in (
            "observations.csv",
            "photos.csv",
            "taxa.csv",
            "observers.csv",
        ):
            payload = f"header\n{name}\n".encode("utf-8")
            expected[f"{name}.gz"] = payload
            info = tarfile.TarInfo(f"monthly-export/{name}")
            info.size = len(payload)
            bundle.addfile(info, BytesIO(payload))
        ignored = b"not metadata"
        info = tarfile.TarInfo("monthly-export/projects.csv")
        info.size = len(ignored)
        bundle.addfile(info, BytesIO(ignored))

    out = tmp_path / "out"
    paths = extract_selected(archive, out)

    assert {path.name for path in paths} == set(expected)
    for path in paths:
        with gzip.open(path, "rb") as handle:
            assert handle.read() == expected[path.name]
    assert not (out / "projects.csv.gz").exists()


def test_multidomain_weights_and_topk():
    labels = np.array(["A", "A", "A", "B"])
    sources = np.array(["inat", "inat", "inat", "bioscan"])
    weights = balanced_weights(labels, sources)
    assert weights[-1] > weights[0]
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])
    assert top_k_accuracy(np.array(["A", "B"]), probabilities, np.array(["A", "B"]), 1) == 1.0


def test_colab_notebook_is_valid_json():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/Diptera_Training_Colab.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) >= 5
    micro = json.loads((root / "notebooks/MicroDiptera_v04_Colab.ipynb").read_text(encoding="utf-8"))
    assert micro["nbformat"] == 4
    assert any("train_hierarchical.py" in "".join(cell.get("source", [])) for cell in micro["cells"])
    setup = json.loads((root / "notebooks/Foundation_Corpus_v05_SETUP_Colab.ipynb").read_text(encoding="utf-8"))
    train = json.loads((root / "notebooks/Foundation_Corpus_v05_TRAIN_Colab.ipynb").read_text(encoding="utf-8"))
    bioscan = json.loads((root / "notebooks/BIOSCAN_30K_SELECTIVE_v06_Colab.ipynb").read_text(encoding="utf-8"))
    assert any("download_dissco.py" in "".join(cell.get("source", [])) for cell in setup["cells"])
    assert any("run_bioscan_30k.py" in "".join(cell.get("source", [])) for cell in setup["cells"])
    assert any("run_foundation_v06.py" in "".join(cell.get("source", [])) for cell in train["cells"])
    assert any("--selection-only" in "".join(cell.get("source", [])) for cell in bioscan["cells"])
    assert any("download_report['complete'] == 30000" in "".join(cell.get("source", [])) for cell in bioscan["cells"])


def test_multidomain_trainer_writes_real_models(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    labels = ["Syrphidae"] * 6 + ["Muscidae"] * 6
    splits = ["train"] * 4 + ["val"] * 2 + ["train"] * 4 + ["val"] * 2
    frame = pd.DataFrame({
        "family": labels,
        "genus": ["Eristalis"] * 6 + ["Musca"] * 6,
        "species": ["Eristalis tenax"] * 6 + ["Musca domestica"] * 6,
        "source": ["iNaturalist", "BIOSCAN-5M"] * 6,
        "split": splits,
        "split_group": [f"g-{index}" for index in range(12)],
    })
    frame.to_csv(tmp_path / "embedded_manifest.csv", index=False)
    vectors = np.vstack([
        np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (6, 1)),
        np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (6, 1)),
    ])
    np.save(tmp_path / "embeddings.npy", vectors)
    subprocess.run([
        sys.executable, str(root / "scripts/train_multidomain.py"),
        "--model-dir", str(tmp_path), "--min-images-per-class", "2",
    ], cwd=root, check=True)
    payload = joblib.load(tmp_path / "classifiers_multidomain.joblib")
    assert set(payload["models"]) == {"family", "genus", "species"}
    assert payload["metadata"]["evaluation"]["family"]["top1_accuracy"] == 1.0
