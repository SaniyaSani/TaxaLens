#!/usr/bin/env python3
"""Split a training plan into deterministic, specimen-group-safe embedding shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diptera_id.corpus.io import ManifestWriter, load_manifest


def shard_for(group: str, count: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\x1f{group}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % count


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", default="data/foundation_v05/shards")
    parser.add_argument("--shard-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.shard_size < 1:
        raise SystemExit("shard-size must be positive")

    frame = load_manifest(args.input).fillna("")
    if frame.empty:
        raise SystemExit("training plan is empty")
    group_column = next((name for name in ("duplicate_group_id", "specimen_group_id", "split_group", "record_id") if name in frame), "record_id")
    groups = frame[group_column].astype(str)
    if group_column == "duplicate_group_id":
        groups = groups.where(groups.ne(""), frame.get("specimen_group_id", frame["record_id"]).astype(str))
    group_sizes = groups.value_counts().to_dict()
    shard_count = min(len(group_sizes), max(1, math.ceil(len(frame) / args.shard_size)))
    loads = [0] * shard_count
    group_assignments: dict[str, int] = {}
    ordered_groups = sorted(
        group_sizes,
        key=lambda value: (-group_sizes[value], shard_for(value, 2**31 - 1, args.seed)),
    )
    for group in ordered_groups:
        shard_id = min(range(shard_count), key=lambda value: (loads[value], value))
        group_assignments[group] = shard_id
        loads[shard_id] += int(group_sizes[group])
    assignments = groups.map(group_assignments)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)
    input_sha256 = file_sha256(input_path)
    index = {"input": str(args.input), "input_sha256": input_sha256, "rows": int(len(frame)), "shard_count": shard_count, "group_column": group_column, "shards": []}
    expected_paths: set[Path] = set()
    for shard_id in range(shard_count):
        selected = frame[assignments.eq(shard_id)].copy()
        path = out_dir / f"shard_{shard_id:05d}.parquet"
        expected_paths.add(path)
        with ManifestWriter(path) as writer:
            writer.write(selected)
        index["shards"].append({"id": shard_id, "path": str(path), "rows": int(len(selected)),
                                "sha256": file_sha256(path)})
    # Remove only stale derived manifest shards after every new shard succeeded.
    for stale in set(out_dir.glob("shard_*.parquet")) - expected_paths:
        stale.unlink()
    index_path = out_dir / "shards.json"
    pending = index_path.with_name(index_path.name + f".{uuid.uuid4().hex}.pending")
    pending.write_text(json.dumps(index, indent=2), encoding="utf-8")
    pending.replace(index_path)
    sizes = [item["rows"] for item in index["shards"]]
    print(f"wrote {shard_count} group-safe shards; rows min={min(sizes)} max={max(sizes)} -> {out_dir}")


if __name__ == "__main__":
    main()
