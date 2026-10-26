#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-gbif-recovery
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=06:00:00
#SBATCH --array=0-23%5
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"
IMAGE_ROOT="${IMAGE_ROOT:-/scratch/$USER/taxalens/image_cache}"

DOWNLOAD_DIR="$DATA_ROOT/pipeline/image_download"
AUDIT_DIR="$DOWNLOAD_DIR/gbif_recovery_v1/laji_replacement_audit"
V3="$AUDIT_DIR/laji_replacement_candidates_v3.parquet"
V4="$AUDIT_DIR/cross_source_v4/cross_source_candidates_v4.parquet"

GBIF_PLAN="$DOWNLOAD_DIR/gbif_plan.parquet"
GBIF_MISSING="$DOWNLOAD_DIR/gbif_recovery_v1/gbif_missing.parquet"

RECOVERY_ROOT="$DOWNLOAD_DIR/gbif_parallel_recovery_v1"
PLAN_DIR="$RECOVERY_ROOT/plans"
CACHED_DIR="$RECOVERY_ROOT/cached"
REPORT_DIR="$RECOVERY_ROOT/reports"
JOB_TMP="$DATA_ROOT/tmp/gbif-recovery-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p \
    "$PLAN_DIR" \
    "$CACHED_DIR" \
    "$REPORT_DIR" \
    "$JOB_TMP"

export TMPDIR="$JOB_TMP"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

TASK_ID="$SLURM_ARRAY_TASK_ID"

if (( TASK_ID < 16 ))
then
    MODE="replacement"
    LOCAL_SHARD="$TASK_ID"
    SHARD_COUNT=16
    SOURCE_MANIFEST="$PLAN_DIR/replacement_part_$(printf '%02d' "$LOCAL_SHARD").parquet"
    CACHED_MANIFEST="$CACHED_DIR/replacement_cached_part_$(printf '%02d' "$LOCAL_SHARD").parquet"
    SHARD_REPORT="$REPORT_DIR/replacement_part_$(printf '%02d' "$LOCAL_SHARD").json"

    INPUT_A="$V3"
    INPUT_B="$V4"

    WORKERS=6
    SLEEP=0.20
else
    MODE="original_retry"
    LOCAL_SHARD=$((TASK_ID - 16))
    SHARD_COUNT=8
    SOURCE_MANIFEST="$PLAN_DIR/original_retry_part_$(printf '%02d' "$LOCAL_SHARD").parquet"
    CACHED_MANIFEST="$CACHED_DIR/original_retry_cached_part_$(printf '%02d' "$LOCAL_SHARD").parquet"
    SHARD_REPORT="$REPORT_DIR/original_retry_part_$(printf '%02d' "$LOCAL_SHARD").json"

    INPUT_A="$GBIF_PLAN"
    INPUT_B="$GBIF_MISSING"

    WORKERS=4
    SLEEP=0.40
fi

echo "Compute node: $(hostname)"
echo "Array job: $SLURM_ARRAY_JOB_ID"
echo "Task: $TASK_ID"
echo "Mode: $MODE"
echo "Local shard: $LOCAL_SHARD/$SHARD_COUNT"
echo "Source manifest: $SOURCE_MANIFEST"
echo "Cached manifest: $CACHED_MANIFEST"
echo

for file in "$INPUT_A" "$INPUT_B"
do
    if [[ ! -s "$file" ]]
    then
        echo "STOP: required input is missing or empty: $file"
        exit 1
    fi
done

echo "=== PREPARE DETERMINISTIC SHARD ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - \
    "$MODE" \
    "$INPUT_A" \
    "$INPUT_B" \
    "$LOCAL_SHARD" \
    "$SHARD_COUNT" \
    "$SOURCE_MANIFEST" <<'PY'
import hashlib
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


mode = sys.argv[1]
input_a = Path(sys.argv[2])
input_b = Path(sys.argv[3])
shard_index = int(sys.argv[4])
shard_count = int(sys.argv[5])
out_path = Path(sys.argv[6])


def shard_for(record_id):
    digest = hashlib.sha256(
        str(record_id).encode("utf-8")
    ).hexdigest()
    return int(digest, 16) % shard_count


def write_atomic(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    frame.to_parquet(pending, index=False)
    pending.replace(path)


if mode == "replacement":
    v3 = pd.read_parquet(input_a)
    v4 = pd.read_parquet(input_b)

    if len(v3) != 6_486:
        raise SystemExit(
            f"STOP: expected 6,486 V3 candidates; found {len(v3):,}"
        )

    if len(v4) != 45:
        raise SystemExit(
            f"STOP: expected 45 V4 candidates; found {len(v4):,}"
        )

    frame = pd.concat(
        [v3, v4],
        ignore_index=True,
        sort=False,
    )

    expected = 6_531

else:
    plan = pd.read_parquet(input_a)
    missing = pd.read_parquet(input_b)

    required_missing = {"record_id", "image_url"}
    absent = required_missing - set(missing.columns)

    if absent:
        raise SystemExit(
            "STOP: missing-manifest columns absent: "
            + ", ".join(sorted(absent))
        )

    hosts = (
        missing["image_url"]
        .fillna("")
        .astype(str)
        .map(lambda value: (urlparse(value).hostname or "").lower())
    )

    retry_ids = set(
        missing.loc[
            hosts.ne("image.laji.fi"),
            "record_id",
        ].fillna("").astype(str)
    )

    frame = plan[
        plan["record_id"].fillna("").astype(str).isin(retry_ids)
    ].copy()

    expected = 1_885

if len(frame) != expected:
    raise SystemExit(
        f"STOP: expected {expected:,} {mode} rows; "
        f"found {len(frame):,}"
    )

required = {
    "record_id",
    "source",
    "image_url",
    "image_license",
}

absent = required - set(frame.columns)

if absent:
    raise SystemExit(
        "STOP: recovery manifest is missing columns: "
        + ", ".join(sorted(absent))
    )

record_ids = frame["record_id"].fillna("").astype(str)
urls = frame["image_url"].fillna("").astype(str)

if record_ids.eq("").any():
    raise SystemExit("STOP: empty recovery record_id")

if record_ids.duplicated().any():
    raise SystemExit("STOP: duplicate recovery record_id")

if urls.eq("").any():
    raise SystemExit("STOP: empty recovery image_url")

if urls.duplicated().any():
    raise SystemExit("STOP: duplicate recovery image_url")

assignments = record_ids.map(shard_for)
selected = frame[assignments.eq(shard_index)].copy()

if selected.empty:
    raise SystemExit(
        f"STOP: deterministic shard {shard_index} is empty"
    )

write_atomic(selected, out_path)

print(
    f"Prepared {mode} shard {shard_index}/{shard_count}: "
    f"{len(selected):,} rows -> {out_path}"
)
print(selected["source"].value_counts().to_string())
PY

echo
echo "=== DOWNLOAD SHARD ==="
echo "Workers: $WORKERS"
echo "Request interval: $SLEEP seconds"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/download_images.py \
    --manifest "$SOURCE_MANIFEST" \
    --out "$CACHED_MANIFEST" \
    --image-root "$IMAGE_ROOT" \
    --max-images 0 \
    --allow-unbounded \
    --workers "$WORKERS" \
    --sleep "$SLEEP" \
    --connect-timeout 30 \
    --read-timeout 90 \
    --retries 2 \
    --retry-backoff 3 \
    --chunksize 500

echo
echo "=== VALIDATE SHARD STRUCTURE ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - \
    "$MODE" \
    "$LOCAL_SHARD" \
    "$SOURCE_MANIFEST" \
    "$CACHED_MANIFEST" \
    "$SHARD_REPORT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


mode = sys.argv[1]
shard = int(sys.argv[2])
source_path = Path(sys.argv[3])
cached_path = Path(sys.argv[4])
report_path = Path(sys.argv[5])

source = pd.read_parquet(source_path)
cached = pd.read_parquet(cached_path)

if len(source) != len(cached):
    raise SystemExit(
        f"STOP: source/cached row mismatch: "
        f"{len(source):,} != {len(cached):,}"
    )

ids = cached["record_id"].fillna("").astype(str)

if ids.eq("").any() or ids.duplicated().any():
    raise SystemExit("STOP: invalid record IDs in cached shard")

local_paths = cached["local_path"].fillna("").astype(str)

valid = local_paths.map(
    lambda value: bool(value)
    and Path(value).is_file()
    and Path(value).stat().st_size > 0
)

reasons = Counter(
    cached.loc[~valid, "exclusion_reason"]
    .fillna("unknown")
    .astype(str)
)

report = {
    "schema": 1,
    "mode": mode,
    "shard": shard,
    "rows": len(cached),
    "valid_images": int(valid.sum()),
    "missing_images": int((~valid).sum()),
    "failure_reasons": {
        str(key): int(value)
        for key, value in reasons.items()
    },
    "source_counts": {
        str(key): int(value)
        for key, value in cached["source"].value_counts().items()
    },
}

report_path.parent.mkdir(parents=True, exist_ok=True)
pending = report_path.with_name(report_path.name + ".pending")
pending.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
pending.replace(report_path)

print(json.dumps(report, indent=2, ensure_ascii=False))
PY

echo
echo "GBIF recovery shard completed"
