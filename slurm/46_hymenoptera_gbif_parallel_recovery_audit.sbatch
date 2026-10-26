#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-gbif-recov-audit
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"

RECOVERY_ROOT="$DATA_ROOT/pipeline/image_download/gbif_parallel_recovery_v1"
CACHED_DIR="$RECOVERY_ROOT/cached"
FINAL_DIR="$RECOVERY_ROOT/final"

mkdir -p "$FINAL_DIR"
cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Recovery root: $RECOVERY_ROOT"
echo "No training-plan files will be modified"
echo

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - "$CACHED_DIR" "$FINAL_DIR" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


cached_dir = Path(sys.argv[1])
final_dir = Path(sys.argv[2])


def atomic_parquet(frame, path):
    pending = path.with_name(path.name + ".pending")
    frame.to_parquet(pending, index=False)
    pending.replace(path)


def atomic_json(payload, path):
    pending = path.with_name(path.name + ".pending")
    pending.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


replacement_files = [
    cached_dir / f"replacement_cached_part_{index:02d}.parquet"
    for index in range(16)
]

retry_files = [
    cached_dir / f"original_retry_cached_part_{index:02d}.parquet"
    for index in range(8)
]

all_expected = replacement_files + retry_files
missing_files = [
    str(path)
    for path in all_expected
    if not path.is_file() or path.stat().st_size == 0
]

if missing_files:
    report = {
        "schema": 1,
        "complete": False,
        "missing_shard_files": missing_files,
        "training_plan_modified": False,
    }

    atomic_json(
        report,
        final_dir / "parallel_recovery_audit.json",
    )

    raise SystemExit(
        "STOP: missing cached shard manifests: "
        + ", ".join(missing_files)
    )


def merge(paths, expected, label):
    frames = [pd.read_parquet(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True, sort=False)

    if len(frame) != expected:
        raise SystemExit(
            f"STOP: expected {expected:,} {label} rows; "
            f"found {len(frame):,}"
        )

    ids = frame["record_id"].fillna("").astype(str)

    if ids.eq("").any() or ids.duplicated().any():
        raise SystemExit(
            f"STOP: invalid or duplicate {label} record IDs"
        )

    return frame


def split_valid(frame):
    local_paths = frame["local_path"].fillna("").astype(str)

    mask = local_paths.map(
        lambda value: bool(value)
        and Path(value).is_file()
        and Path(value).stat().st_size > 0
    )

    return frame[mask].copy(), frame[~mask].copy()


replacements = merge(
    replacement_files,
    6_531,
    "replacement",
)

retries = merge(
    retry_files,
    1_885,
    "original-retry",
)

overlap = set(replacements["record_id"]) & set(retries["record_id"])

if overlap:
    raise SystemExit(
        f"STOP: replacement/retry overlap: {len(overlap):,} rows"
    )

replacement_valid, replacement_missing = split_valid(replacements)
retry_valid, retry_missing = split_valid(retries)

atomic_parquet(
    replacements,
    final_dir / "replacement_cached_merged.parquet",
)
atomic_parquet(
    replacement_valid,
    final_dir / "replacement_valid.parquet",
)
atomic_parquet(
    replacement_missing,
    final_dir / "replacement_missing.parquet",
)

atomic_parquet(
    retries,
    final_dir / "original_retry_cached_merged.parquet",
)
atomic_parquet(
    retry_valid,
    final_dir / "original_retry_valid.parquet",
)
atomic_parquet(
    retry_missing,
    final_dir / "original_retry_missing.parquet",
)


def reasons(frame):
    if frame.empty:
        return {}

    return {
        str(key): int(value)
        for key, value in Counter(
            frame["exclusion_reason"]
            .fillna("unknown")
            .astype(str)
        ).items()
    }


report = {
    "schema": 1,
    "complete": True,
    "replacement_rows": len(replacements),
    "replacement_valid": len(replacement_valid),
    "replacement_missing": len(replacement_missing),
    "replacement_valid_fraction": (
        len(replacement_valid) / len(replacements)
    ),
    "replacement_sources": {
        str(key): int(value)
        for key, value in replacements["source"].value_counts().items()
    },
    "replacement_failure_reasons": reasons(replacement_missing),
    "original_retry_rows": len(retries),
    "original_retry_valid": len(retry_valid),
    "original_retry_missing": len(retry_missing),
    "original_retry_valid_fraction": (
        len(retry_valid) / len(retries)
    ),
    "original_retry_failure_reasons": reasons(retry_missing),
    "training_plan_modified": False,
}

atomic_json(
    report,
    final_dir / "parallel_recovery_audit.json",
)

print(json.dumps(report, indent=2, ensure_ascii=False))
print()
print(
    "Report ->",
    final_dir / "parallel_recovery_audit.json",
)
PY

echo
echo "=== FINAL RECOVERY OUTPUTS ==="

find "$FINAL_DIR" \
    -maxdepth 1 \
    -type f \
    -printf '%12s bytes  %f\n' |
    sort

echo
echo "Parallel GBIF recovery audit completed"
echo "No training-plan files were modified"
