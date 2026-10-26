#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-gbif-audit
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"

SHARD_ROOT="$DATA_ROOT/pipeline/image_download/gbif_shards_v1"
OUT_DIR="$DATA_ROOT/pipeline/image_download/gbif_recovery_v1"

mkdir -p "$OUT_DIR"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Shard root: $SHARD_ROOT"
echo "Output directory: $OUT_DIR"
echo

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - "$SHARD_ROOT" "$OUT_DIR" <<'PYAUDIT'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

shard_root = Path(sys.argv[1])
out_dir = Path(sys.argv[2])

plan_paths = sorted(
    (shard_root / "plans").glob("gbif_part_*.parquet")
)
cached_paths = sorted(
    (shard_root / "cached").glob("gbif_cached_part_*.parquet")
)

if len(plan_paths) != 8:
    raise SystemExit(
        f"STOP: expected 8 plan shards, found {len(plan_paths)}"
    )

if len(cached_paths) != 8:
    raise SystemExit(
        f"STOP: expected 8 cached shards, found {len(cached_paths)}"
    )

print("=== LOAD AND VERIFY SHARDS ===")

plan_frames = []
cached_frames = []

for plan_path, cached_path in zip(plan_paths, cached_paths):
    plan = pd.read_parquet(plan_path)
    cached = pd.read_parquet(cached_path)

    plan_ids = set(plan["record_id"].astype(str))
    cached_ids = set(cached["record_id"].astype(str))

    if len(plan) != len(cached):
        raise SystemExit(
            f"STOP: row mismatch for {cached_path.name}: "
            f"{len(plan):,} plan vs {len(cached):,} cached"
        )

    if plan_ids != cached_ids:
        raise SystemExit(
            f"STOP: record IDs differ for {cached_path.name}"
        )

    print(
        f"{cached_path.name}: "
        f"{len(cached):,} rows verified"
    )

    plan_frames.append(plan)
    cached_frames.append(cached)

plan = pd.concat(plan_frames, ignore_index=True)
cached = pd.concat(cached_frames, ignore_index=True)

if len(plan) != 85_000 or len(cached) != 85_000:
    raise SystemExit(
        f"STOP: expected 85,000 rows; "
        f"plan={len(plan):,}, cached={len(cached):,}"
    )

if plan["record_id"].astype(str).duplicated().any():
    raise SystemExit("STOP: duplicate record IDs in merged plan")

if cached["record_id"].astype(str).duplicated().any():
    raise SystemExit("STOP: duplicate record IDs in merged cache")

if set(plan["record_id"].astype(str)) != set(
    cached["record_id"].astype(str)
):
    raise SystemExit(
        "STOP: merged plan and cache record IDs differ"
    )

print()
print("=== CHECK LOCAL IMAGE FILES ===")

def valid_file(value: object) -> bool:
    text = "" if value is None else str(value).strip()

    if not text:
        return False

    path = Path(text)

    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False

cached["image_available"] = (
    cached["local_path"]
    .fillna("")
    .map(valid_file)
)

valid = cached[cached["image_available"]].copy()
missing = cached[~cached["image_available"]].copy()

if len(valid) + len(missing) != 85_000:
    raise SystemExit("STOP: audit row accounting mismatch")

if len(valid) != 76_540:
    raise SystemExit(
        f"STOP: expected 76,540 valid images, "
        f"found {len(valid):,}"
    )

if len(missing) != 8_460:
    raise SystemExit(
        f"STOP: expected 8,460 missing images, "
        f"found {len(missing):,}"
    )

def host(value: object) -> str:
    text = "" if value is None else str(value).strip()

    if not text:
        return "__missing_url__"

    try:
        return urlparse(text).netloc.lower() or "__unknown__"
    except ValueError:
        return "__invalid_url__"

missing["image_host"] = (
    missing["image_url"]
    .fillna("")
    .map(host)
)

reasons = (
    missing.get(
        "exclusion_reason",
        pd.Series("", index=missing.index),
    )
    .fillna("")
    .astype(str)
    .replace("", "missing_or_empty_local_file")
)

def counts(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.value_counts().items()
    }

family_counts = counts(
    missing["family"]
    .fillna("__unresolved__")
    .astype(str)
)

top_missing_families = dict(
    list(family_counts.items())[:30]
)

report = {
    "schema": 1,
    "rows": len(cached),
    "valid_images": len(valid),
    "missing_images": len(missing),
    "valid_fraction": len(valid) / len(cached),
    "unique_record_ids": int(
        cached["record_id"].astype(str).nunique()
    ),
    "missing_by_reason": counts(reasons),
    "missing_by_host": counts(missing["image_host"]),
    "missing_by_split": counts(
        missing["split"].fillna("__missing__").astype(str)
    ),
    "top_missing_families": top_missing_families,
    "missing_unique_urls": int(
        missing["image_url"]
        .fillna("")
        .astype(str)
        .replace("", pd.NA)
        .nunique()
    ),
}

out_dir.mkdir(parents=True, exist_ok=True)

outputs = {
    "merged": (
        out_dir / "gbif_cached_merged.parquet",
        cached,
    ),
    "valid": (
        out_dir / "gbif_valid.parquet",
        valid,
    ),
    "missing": (
        out_dir / "gbif_missing.parquet",
        missing,
    ),
}

for label, (destination, frame) in outputs.items():
    pending = destination.with_name(
        destination.name + ".pending"
    )
    frame.to_parquet(pending, index=False)
    pending.replace(destination)
    print(
        f"{label}: {len(frame):,} rows -> {destination}"
    )

report_path = out_dir / "gbif_failure_audit.json"
pending_report = report_path.with_name(
    report_path.name + ".pending"
)
pending_report.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
pending_report.replace(report_path)

print()
print(json.dumps(report, indent=2, ensure_ascii=False))
print()
print(f"Audit report -> {report_path}")
PYAUDIT

echo
echo "=== AUDIT OUTPUTS ==="

ls -lh \
    "$OUT_DIR/gbif_cached_merged.parquet" \
    "$OUT_DIR/gbif_valid.parquet" \
    "$OUT_DIR/gbif_missing.parquet" \
    "$OUT_DIR/gbif_failure_audit.json"

echo
echo "GBIF failure audit completed"
echo "No training-plan files or cached images were modified"
