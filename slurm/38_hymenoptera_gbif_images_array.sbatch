#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-gbif-shard
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-7%4
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"
IMAGE_ROOT="${IMAGE_ROOT:-/scratch/$USER/taxalens/image_cache}"

PLAN="$DATA_ROOT/pipeline/image_download/gbif_plan.parquet"
SHARD_ROOT="$DATA_ROOT/pipeline/image_download/gbif_shards_v1"

SHARD_COUNT=8
SHARD_ID="$SLURM_ARRAY_TASK_ID"
TAG=$(printf '%02d' "$SHARD_ID")

SOURCE_MANIFEST="$SHARD_ROOT/plans/gbif_part_${TAG}.parquet"
CACHED_MANIFEST="$SHARD_ROOT/cached/gbif_cached_part_${TAG}.parquet"
JOB_TMP="$DATA_ROOT/tmp/gbif-images-${SLURM_ARRAY_JOB_ID}-${SHARD_ID}"

mkdir -p \
    "$SHARD_ROOT/plans" \
    "$SHARD_ROOT/cached" \
    "$IMAGE_ROOT" \
    "$JOB_TMP"

export TMPDIR="$JOB_TMP"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Array job: $SLURM_ARRAY_JOB_ID"
echo "Shard: $SHARD_ID/$SHARD_COUNT"
echo "Training-plan subset: $PLAN"
echo "Shared image cache: $IMAGE_ROOT"
echo

if [[ ! -s "$PLAN" ]]
then
    echo "STOP: GBIF plan is missing or empty: $PLAN"
    exit 1
fi

echo "=== PREPARE DETERMINISTIC GBIF SHARD ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - "$PLAN" "$SOURCE_MANIFEST" "$SHARD_ID" "$SHARD_COUNT" <<'PYSHARD'
import hashlib
import sys
from pathlib import Path

import pandas as pd

plan_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
shard_id = int(sys.argv[3])
shard_count = int(sys.argv[4])

frame = pd.read_parquet(plan_path)

required = {"record_id", "source", "image_url", "image_license"}
missing = required - set(frame.columns)

if missing:
    raise SystemExit(
        "STOP: GBIF plan is missing columns: "
        + ", ".join(sorted(missing))
    )

frame = frame[
    frame["source"].fillna("").astype(str).eq("GBIF")
].copy()

if len(frame) != 85_000:
    raise SystemExit(
        f"STOP: expected 85,000 GBIF rows, found {len(frame):,}"
    )

record_ids = frame["record_id"].fillna("").astype(str)

if record_ids.eq("").any():
    raise SystemExit("STOP: empty GBIF record_id")

if record_ids.duplicated().any():
    raise SystemExit("STOP: duplicate GBIF record_id")

def assigned_shard(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count

mask = record_ids.map(assigned_shard).eq(shard_id)
selected = frame.loc[mask].copy()

if selected.empty:
    raise SystemExit(f"STOP: shard {shard_id} is empty")

out_path.parent.mkdir(parents=True, exist_ok=True)
pending = out_path.with_name(out_path.name + ".pending")
selected.to_parquet(pending, index=False)
pending.replace(out_path)

print(
    f"Prepared GBIF shard {shard_id}/{shard_count}: "
    f"{len(selected):,} rows -> {out_path}"
)
PYSHARD

echo
echo "=== DOWNLOAD GBIF SHARD ==="
echo "Existing valid files will be verified and skipped"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/download_images.py \
    --manifest "$SOURCE_MANIFEST" \
    --out "$CACHED_MANIFEST" \
    --image-root "$IMAGE_ROOT" \
    --max-images 0 \
    --allow-unbounded \
    --workers 4 \
    --sleep 0.25 \
    --chunksize 500

echo
echo "=== VALIDATE GBIF SHARD ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - "$SOURCE_MANIFEST" "$CACHED_MANIFEST" <<'PYVALIDATE'
import sys
from pathlib import Path

import pandas as pd

source_path = Path(sys.argv[1])
cached_path = Path(sys.argv[2])

if not cached_path.is_file() or cached_path.stat().st_size == 0:
    raise SystemExit("STOP: cached shard manifest is missing or empty")

source = pd.read_parquet(source_path)
cached = pd.read_parquet(cached_path)

if len(cached) != len(source):
    raise SystemExit(
        f"STOP: expected {len(source):,} cached rows, "
        f"found {len(cached):,}"
    )

record_ids = cached["record_id"].fillna("").astype(str)

if record_ids.duplicated().any():
    raise SystemExit("STOP: duplicate record_id in cached shard")

local_paths = cached["local_path"].fillna("").astype(str)

missing = sum(
    not value
    or not Path(value).is_file()
    or Path(value).stat().st_size == 0
    for value in local_paths
)

print(f"Shard rows: {len(cached):,}")
print(f"Valid images: {len(cached) - missing:,}")
print(f"Missing images: {missing:,}")
print(f"Cached manifest: {cached_path}")

if missing:
    raise SystemExit(
        f"STOP: {missing:,} GBIF images remain unavailable; "
        "the completed manifest was preserved for retry/replacement"
    )
PYVALIDATE

echo
echo "GBIF shard $SHARD_ID completed successfully"
