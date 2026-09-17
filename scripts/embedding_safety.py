"""Shared checkpoint and specimen-alignment checks; no model downloads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


KEY_COLUMNS = ("specimen_group_id", "record_id", "split_group")
COHORT_COLUMNS = (
    "record_id", "specimen_group_id", "split_group", "split", "source",
    "source_split", "family", "genus", "species", "label_quality",
    "eligible_family", "eligible_genus", "eligible_species", "view_count",
    "available_views", "embedding_view_ids",
)


def digest(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def semantic_config(config: dict) -> dict:
    """Storage format and the number of views can legitimately vary by shard."""
    return {
        "backend": config.get("backend", "dino"),
        "backbone": config.get("backbone", config.get("model")),
        "image_size": int(config.get("image_size", 512)),
        "tile_grid": int(config.get("tile_grid", 1)),
        "include_whole": bool(config.get("include_whole", True)),
        "local_only": bool(config.get("local_only", False)),
        "preprocessing": str(config.get("preprocessing", "square_pad_v1")),
    }


def checkpoint_ready(complete: Path, manifest: Path, config: dict) -> bool:
    try:
        saved = json.loads(complete.read_text(encoding="utf-8"))
        if saved.get("manifest_sha256") != digest(manifest):
            return False
        if semantic_config(saved["embedding"]) != semantic_config(config):
            return False
        # A marker alone is insufficient: arrays may have been interrupted or removed.
        hashes = saved.get("artifact_sha256", {})
        return all(
            hashes.get(name) == digest(complete.parent / name)
            for name in ("embeddings.npy", "embedded_manifest.csv")
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def common_key_column(*frames) -> str:
    for column in KEY_COLUMNS:
        if all(column in frame and frame[column].astype(str).str.strip().ne("").all()
               for frame in frames):
            if any(frame[column].astype(str).duplicated().any() for frame in frames):
                raise SystemExit(f"duplicate specimen keys in {column}; repair grouping before fusion")
            return column
    raise SystemExit("no common complete specimen key; cannot align encoders safely")


def check_aligned_metadata(left, right) -> None:
    for column in COHORT_COLUMNS:
        if column not in left and column not in right:
            continue
        if column not in left or column not in right:
            raise SystemExit(f"encoder metadata missing column: {column}")
        if left[column].astype(str).tolist() != right[column].astype(str).tolist():
            raise SystemExit(f"encoder metadata differs for matched specimens: {column}")


def cohort_indices(frame, cohort) -> list[int]:
    column = common_key_column(frame, cohort)
    lookup = {str(key): index for index, key in enumerate(frame[column])}
    missing = set(cohort[column].astype(str)) - lookup.keys()
    if missing:
        raise SystemExit(f"model is missing {len(missing)} specimens from the shared cohort")
    indices = [lookup[str(key)] for key in cohort[column]]
    check_aligned_metadata(frame.iloc[indices], cohort)
    return indices


def cohort_digest(frame) -> str:
    column = common_key_column(frame)
    columns = sorted(set([column]) | (set(COHORT_COLUMNS) & set(frame.columns)))
    normalized = frame.sort_values(column)[columns].astype(str)
    encoded = json.dumps({"columns": columns, "rows": normalized.values.tolist()},
                         ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
