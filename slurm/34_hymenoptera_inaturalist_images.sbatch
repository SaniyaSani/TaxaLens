#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-inat-images
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"
IMAGE_ROOT="${IMAGE_ROOT:-/scratch/$USER/taxalens/image_cache}"

PLAN="$DATA_ROOT/pipeline/planning/training_plan_v1/training_plan.parquet"
DOWNLOAD_DIR="$DATA_ROOT/pipeline/image_download"
SOURCE_MANIFEST="$DOWNLOAD_DIR/inaturalist_plan.parquet"
CACHED_MANIFEST="$DOWNLOAD_DIR/inaturalist_cached.parquet"
JOB_TMP="$DATA_ROOT/tmp/inaturalist-images-${SLURM_JOB_ID}"

mkdir -p \
    "$DOWNLOAD_DIR" \
    "$IMAGE_ROOT" \
    "$JOB_TMP"

export TMPDIR="$JOB_TMP"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Training plan: $PLAN"
echo "Shared image cache: $IMAGE_ROOT"
echo

if [[ ! -s "$PLAN" ]]
then
    echo "STOP: training plan is missing or empty: $PLAN"
    exit 1
fi

echo "=== PREPARE INATURALIST MANIFEST ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - "$PLAN" "$SOURCE_MANIFEST" <<'PYFILTER'
import sys
from pathlib import Path

import pandas as pd

plan_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

frame = pd.read_parquet(plan_path)
required = {"record_id", "source", "image_url", "image_license"}
missing = required - set(frame.columns)

if missing:
    raise SystemExit(
        "STOP: training plan is missing columns: "
        + ", ".join(sorted(missing))
    )

selected = frame[
    frame["source"].fillna("").astype(str).eq("iNaturalist")
].copy()

expected = 60_885

if len(selected) != expected:
    raise SystemExit(
        f"STOP: expected {expected:,} iNaturalist rows, "
        f"found {len(selected):,}"
    )

if selected["record_id"].fillna("").astype(str).eq("").any():
    raise SystemExit("STOP: empty iNaturalist record_id")

if selected["record_id"].astype(str).duplicated().any():
    raise SystemExit("STOP: duplicate iNaturalist record_id")

if selected["image_url"].fillna("").astype(str).eq("").any():
    raise SystemExit("STOP: empty iNaturalist image_url")

out_path.parent.mkdir(parents=True, exist_ok=True)
pending = out_path.with_name(out_path.name + ".pending")
selected.to_parquet(pending, index=False)
pending.replace(out_path)

print(f"Prepared {len(selected):,} iNaturalist rows -> {out_path}")
PYFILTER

ls -lh "$SOURCE_MANIFEST"

echo
echo "=== DOWNLOAD INATURALIST IMAGES ==="
echo "Existing valid files will be verified and skipped"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/download_images.py \
    --manifest "$SOURCE_MANIFEST" \
    --out "$CACHED_MANIFEST" \
    --image-root "$IMAGE_ROOT" \
    --max-images 0 \
    --allow-unbounded

echo
echo "=== VALIDATE CACHED MANIFEST ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - "$CACHED_MANIFEST" <<'PYVALIDATE'
import sys
from pathlib import Path

import pandas as pd

path = Path(sys.argv[1])

if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit("STOP: cached manifest is missing or empty")

frame = pd.read_parquet(path)

if len(frame) != 60_885:
    raise SystemExit(
        f"STOP: expected 60,885 cached rows, found {len(frame):,}"
    )

local_paths = frame["local_path"].fillna("").astype(str)

missing = sum(
    not value or
    not Path(value).is_file() or
    Path(value).stat().st_size == 0
    for value in local_paths
)

if missing:
    raise SystemExit(
        f"STOP: {missing:,} iNaturalist rows do not have "
        "a non-empty cached image; rerun this job to resume"
    )

print(f"Validated iNaturalist cached rows: {len(frame):,}")
print(f"Unique record IDs: {frame['record_id'].astype(str).nunique():,}")
print(f"Cached manifest: {path}")
PYVALIDATE

echo
echo "=== FINAL OUTPUT ==="

ls -lh \
    "$SOURCE_MANIFEST" \
    "$CACHED_MANIFEST"

du -sh "$IMAGE_ROOT/inaturalist"

echo
echo "iNaturalist Hymenoptera image download completed"
echo "The job is safely resumable by submitting it again"
