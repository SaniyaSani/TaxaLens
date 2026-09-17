#!/usr/bin/env python3
"""Merge completed embedding shards into one model directory without a RAM spike."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from embedding_safety import semantic_config, checkpoint_ready


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--manifest-root", help="When set, merge only shards listed by current manifests")
    parser.add_argument("--out-dir", default="models_foundation_v05")
    args = parser.parse_args()

    if args.manifest_root:
        names = [path.stem for path in sorted(Path(args.manifest_root).glob("shard_*.parquet"))]
        shard_dirs = [Path(args.shard_root) / name for name in names]
        missing = [str(path) for path in shard_dirs if not (path / "complete.json").is_file()]
        if missing:
            raise SystemExit(f"missing completed embedding shards: {len(missing)}")
    else:
        shard_dirs = sorted(path.parent for path in Path(args.shard_root).glob("shard_*/complete.json"))
    if not shard_dirs:
        raise SystemExit("no completed embedding shards found")
    arrays = []
    configs = []
    total = 0
    dimension = None
    for shard in shard_dirs:
        saved = json.loads((shard / "complete.json").read_text(encoding="utf-8"))
        if args.manifest_root and not checkpoint_ready(
            shard / "complete.json", Path(args.manifest_root) / f"{shard.name}.parquet", saved["embedding"]
        ):
            raise SystemExit(f"stale or incomplete embedding shard: {shard}")
        array = np.load(shard / "embeddings.npy", mmap_mode="r")
        if array.ndim != 2:
            raise SystemExit(f"invalid embeddings in {shard}")
        dimension = dimension or int(array.shape[1])
        if array.shape[1] != dimension:
            raise SystemExit("embedding dimensions differ between shards")
        arrays.append((shard, array))
        total += int(array.shape[0])
        configs.append(semantic_config(saved["embedding"]))
    if any(config != configs[0] for config in configs[1:]):
        raise SystemExit("embedding configuration differs between completed shards")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = np.lib.format.open_memmap(out_dir / "embeddings.npy", mode="w+", dtype=np.float16, shape=(total, dimension))
    manifest_path = out_dir / "embedded_manifest.csv"
    first = True
    cursor = 0
    for shard, array in arrays:
        frame = pd.read_csv(shard / "embedded_manifest.csv", dtype=str, keep_default_na=False)
        if len(frame) != len(array):
            raise SystemExit(f"manifest/vector mismatch in {shard}")
        merged[cursor:cursor + len(array)] = array.astype(np.float16)
        cursor += len(array)
        frame.to_csv(manifest_path, mode="w" if first else "a", header=first, index=False)
        first = False
    merged.flush()
    config = dict(configs[0])
    config.update({"embedding_shards": len(shard_dirs), "fused_specimens": total, "storage_dtype": "float16"})
    (out_dir / "embedding_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"merged {len(shard_dirs)} shards / {total} specimens / {dimension} dimensions -> {out_dir}")


if __name__ == "__main__":
    main()
