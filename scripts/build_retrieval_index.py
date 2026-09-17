#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    x = np.load(model_dir / "embeddings.npy").astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.clip(norms, 1e-12, None)
    df = pd.read_csv(model_dir / "embedded_manifest.csv")

    np.save(model_dir / "retrieval_vectors.npy", x.astype(np.float16))
    keep_cols = [c for c in [
        "source", "observation_id", "photo_id", "image_url", "local_path",
        "family", "genus", "species", "event_date", "observer", "photo_license",
        "attribution", "observation_url", "specimen_group_id", "view_type",
        "view_count", "available_views", "label_quality"
    ] if c in df.columns]
    df[keep_cols].to_csv(model_dir / "retrieval_metadata.csv", index=False)

    try:
        import faiss
        index = faiss.IndexFlatIP(x.shape[1])
        index.add(x)
        faiss.write_index(index, str(model_dir / "retrieval.index"))
        print("FAISS index written")
    except Exception as exc:
        print(f"FAISS unavailable ({exc}); NumPy cosine-search fallback will be used.")

    print(f"Indexed {len(x)} reference images")


if __name__ == "__main__":
    main()
