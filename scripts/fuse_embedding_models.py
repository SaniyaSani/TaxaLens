#!/usr/bin/env python3
"""Fuse two merged encoder outputs after aligning the same specimens safely."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from embedding_safety import common_key_column, check_aligned_metadata, cohort_digest
except ModuleNotFoundError:  # imported as scripts.fuse_embedding_models in tests
    from scripts.embedding_safety import common_key_column, check_aligned_metadata, cohort_digest


def load_model(path: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    frame = pd.read_csv(path / "embedded_manifest.csv", dtype=str, keep_default_na=False)
    vectors = np.load(path / "embeddings.npy", mmap_mode="r")
    if vectors.ndim != 2 or len(frame) != len(vectors):
        raise SystemExit(f"manifest/vector mismatch in {path}")
    config = json.loads((path / "embedding_config.json").read_text(encoding="utf-8"))
    return frame, vectors, config


def fuse(dino_dir: Path, bioclip_dir: Path, out_dir: Path) -> dict:
    dino_frame, dino_vectors, dino_config = load_model(dino_dir)
    bio_frame, bio_vectors, bio_config = load_model(bioclip_dir)
    key_column = common_key_column(dino_frame, bio_frame)
    dino_keys = dino_frame[key_column].astype(str)
    bio_keys = bio_frame[key_column].astype(str)
    bio_lookup = {key: index for index, key in enumerate(bio_keys)}
    paired = [(index, bio_lookup[key]) for index, key in enumerate(dino_keys) if key in bio_lookup]
    if not paired:
        raise SystemExit("DINO and BioCLIP have no common specimens")

    dino_indices = np.asarray([left for left, _ in paired], dtype=np.int64)
    bio_indices = np.asarray([right for _, right in paired], dtype=np.int64)
    dino = np.asarray(dino_vectors[dino_indices], dtype=np.float32)
    bio = np.asarray(bio_vectors[bio_indices], dtype=np.float32)
    dino /= np.clip(np.linalg.norm(dino, axis=1, keepdims=True), 1e-12, None)
    bio /= np.clip(np.linalg.norm(bio, axis=1, keepdims=True), 1e-12, None)

    # Each encoder contributes unit energy before the final normalization.
    combined = np.concatenate([dino, bio], axis=1) / np.sqrt(2.0)
    combined /= np.clip(np.linalg.norm(combined, axis=1, keepdims=True), 1e-12, None)
    frame = dino_frame.iloc[dino_indices].reset_index(drop=True)
    paired_bio_frame = bio_frame.iloc[bio_indices].reset_index(drop=True)
    check_aligned_metadata(frame, paired_bio_frame)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", combined.astype(np.float16))
    frame.to_csv(out_dir / "embedded_manifest.csv", index=False)
    encoder_config = [
        {"name": "dino", **dino_config},
        {"name": "bioclip", **bio_config},
    ]
    config = {
        "backend": "fusion",
        "method": "equal_weight_l2_concat",
        "encoders": encoder_config,
        "dimension": int(combined.shape[1]),
        "storage_dtype": "float16",
        "fused_specimens": int(len(frame)),
        "dino_only_specimens": int(len(dino_frame) - len(frame)),
        "bioclip_only_specimens": int(len(bio_frame) - len(frame)),
        "specimen_key": key_column,
        "cohort_sha256": cohort_digest(frame),
    }
    (out_dir / "embedding_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-dir", required=True)
    parser.add_argument("--bioclip-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    report = fuse(Path(args.dino_dir), Path(args.bioclip_dir), Path(args.out_dir))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
