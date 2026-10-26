#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-round2-download
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-23%10
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"
IMAGE_ROOT="${IMAGE_ROOT:-/scratch/$USER/taxalens/image_cache}"

ROUND2_DIR="$DATA_ROOT/pipeline/image_download/gbif_round2_candidates_v1"
SOURCE="$ROUND2_DIR/round2_candidates.parquet"
SOURCE_REPORT="$ROUND2_DIR/round2_candidate_audit.json"

OUTPUT_ROOT="$DATA_ROOT/pipeline/image_download/gbif_round2_download_v1"
PLAN_DIR="$OUTPUT_ROOT/plans"
CACHED_DIR="$OUTPUT_ROOT/cached"

SHARD_COUNT=24
SHARD_ID="$SLURM_ARRAY_TASK_ID"
printf -v SHARD_TAG '%02d' "$SHARD_ID"

SHARD_PLAN="$PLAN_DIR/round2_part_${SHARD_TAG}.parquet"
SHARD_CACHED="$CACHED_DIR/round2_cached_part_${SHARD_TAG}.parquet"
JOB_TMP="$DATA_ROOT/tmp/gbif-round2-${SLURM_ARRAY_JOB_ID}-${SHARD_TAG}"

mkdir -p \
    "$PLAN_DIR" \
    "$CACHED_DIR" \
    "$JOB_TMP"

export TMPDIR="$JOB_TMP"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Array job: $SLURM_ARRAY_JOB_ID"
echo "Shard: $SHARD_ID/$SHARD_COUNT"
echo "Candidate source: $SOURCE"
echo

if [[ ! -s "$SOURCE" ]] || [[ ! -s "$SOURCE_REPORT" ]]
then
    echo "STOP: round-2 candidate inputs are missing"
    exit 1
fi

echo "=== PREPARE DETERMINISTIC SHARD ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - \
    "$SOURCE" \
    "$SOURCE_REPORT" \
    "$SHARD_PLAN" \
    "$SHARD_ID" \
    "$SHARD_COUNT" <<'PYSHARD'
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

source = Path(sys.argv[1])
report_path = Path(sys.argv[2])
destination = Path(sys.argv[3])
shard_id = int(sys.argv[4])
shard_count = int(sys.argv[5])

frame = pd.read_parquet(source)
report = json.loads(report_path.read_text(encoding="utf-8"))

expected = int(report["candidate_rows"])

if len(frame) != expected:
    raise SystemExit(
        f"STOP: expected {expected:,} candidates, found {len(frame):,}"
    )

if "record_id" not in frame:
    raise SystemExit("STOP: candidate manifest has no record_id")

record_ids = frame["record_id"].fillna("").astype(str)

if record_ids.eq("").any():
    raise SystemExit("STOP: empty candidate record_id")

if record_ids.duplicated().any():
    raise SystemExit("STOP: duplicate candidate record_id")

def shard_for(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % shard_count

mask = record_ids.map(shard_for).eq(shard_id)
selected = frame.loc[mask].copy()

if selected.empty:
    raise SystemExit(f"STOP: shard {shard_id} is unexpectedly empty")

destination.parent.mkdir(parents=True, exist_ok=True)
pending = destination.with_name(destination.name + ".pending")
selected.to_parquet(pending, index=False)
pending.replace(destination)

print(
    f"Prepared shard {shard_id}/{shard_count}: "
    f"{len(selected):,} rows -> {destination}"
)
PYSHARD

echo
echo "=== DOWNLOAD ROUND-2 SHARD ==="
echo "Existing valid cached images will be verified and skipped"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/download_images.py \
    --manifest "$SHARD_PLAN" \
    --out "$SHARD_CACHED" \
    --image-root "$IMAGE_ROOT" \
    --max-images 0 \
    --allow-unbounded \
    --workers 4 \
    --sleep 0.20 \
    --connect-timeout 20 \
    --read-timeout 60 \
    --retries 2 \
    --retry-backoff 2 \
    --chunksize 500

echo
echo "=== VALIDATE COMPLETED SHARD MANIFEST ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - "$SHARD_PLAN" "$SHARD_CACHED" <<'PYVALIDATE'
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

plan_path = Path(sys.argv[1])
cached_path = Path(sys.argv[2])

if not cached_path.is_file() or cached_path.stat().st_size == 0:
    raise SystemExit("STOP: cached shard manifest is missing or empty")

plan = pd.read_parquet(plan_path)
cached = pd.read_parquet(cached_path)

plan_ids = set(plan["record_id"].astype(str))
cached_ids = set(cached["record_id"].astype(str))

if len(cached) != len(plan):
    raise SystemExit(
        f"STOP: shard row mismatch: plan={len(plan):,}, "
        f"cached={len(cached):,}"
    )

if plan_ids != cached_ids:
    raise SystemExit("STOP: cached shard record IDs differ from shard plan")

local_paths = cached["local_path"].fillna("").astype(str)

valid = local_paths.map(
    lambda value: bool(value)
    and Path(value).is_file()
    and Path(value).stat().st_size > 0
)

reasons = Counter(
    cached.loc[~valid, "exclusion_reason"]
    .fillna("")
    .astype(str)
)

summary = {
    "rows": int(len(cached)),
    "valid_images": int(valid.sum()),
    "missing_images": int((~valid).sum()),
    "failure_reasons": dict(reasons),
}

print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"Completed manifest: {cached_path}")
PYVALIDATE

echo
echo "Round-2 shard $SHARD_TAG completed"
echo "No training-plan files were modified"
