#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-inat-backup
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"
IMAGE_ROOT="/scratch/$USER/taxalens/image_cache/inaturalist"

PLAN_DIR="$DATA_ROOT/pipeline/planning/training_plan_v1"
DOWNLOAD_DIR="$DATA_ROOT/pipeline/image_download"
CACHED="$DOWNLOAD_DIR/inaturalist_cached.parquet"

BACKUP_PARENT="/shares/hawlitschek.ieu.uzh/$USER/TaxaLens/v11_hymenoptera_teacher"
BACKUP="$BACKUP_PARENT/inaturalist_images_v1"
STAGING="$BACKUP_PARENT/.inaturalist_images_v1.pending-${SLURM_JOB_ID}"

WORK_DIR="$DATA_ROOT/tmp/inat-backup-${SLURM_JOB_ID}"
FILE_LIST_NUL="$WORK_DIR/image_files.nul"
FILE_LIST_TXT="$WORK_DIR/IMAGE_FILES.txt"
INPUT_AUDIT="$WORK_DIR/input_audit.json"

EXPECTED_PLAN_SHA="c9138048ecdab8e251e65389a28d7e842ab6d63fafe0e22f2b6aa620882a7a6e"

mkdir -p "$BACKUP_PARENT" "$WORK_DIR"

if [[ -e "$BACKUP" ]]
then
    echo "STOP: permanent checkpoint already exists: $BACKUP"
    exit 1
fi

if [[ -e "$STAGING" ]]
then
    echo "STOP: staging directory already exists: $STAGING"
    exit 1
fi

cd "$REPO_DIR"

echo "=== VALIDATE INPUTS AND BUILD FILE LIST ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" - \
    "$CACHED" \
    "$IMAGE_ROOT" \
    "$PLAN_DIR/training_plan.parquet" \
    "$PLAN_DIR/training_plan_report.json" \
    "$FILE_LIST_NUL" \
    "$FILE_LIST_TXT" \
    "$INPUT_AUDIT" \
    "$EXPECTED_PLAN_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


cached_path = Path(sys.argv[1])
image_root = Path(sys.argv[2]).resolve()
plan_path = Path(sys.argv[3])
report_path = Path(sys.argv[4])
nul_path = Path(sys.argv[5])
text_path = Path(sys.argv[6])
audit_path = Path(sys.argv[7])
expected_plan_sha = sys.argv[8]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


frame = pd.read_parquet(cached_path).fillna("")

if len(frame) != 60_885:
    raise SystemExit(
        f"STOP: expected 60885 cached rows; found {len(frame)}"
    )

if set(frame["source"].astype(str)) != {"iNaturalist"}:
    raise SystemExit("STOP: cached manifest contains another source")

if frame["record_id"].astype(str).duplicated().any():
    raise SystemExit("STOP: cached manifest contains duplicate record IDs")

relative_paths = []
total_bytes = 0

for value in frame["local_path"].astype(str):
    if not value:
        raise SystemExit("STOP: empty local_path in cached manifest")

    source = Path(value).resolve()

    try:
        relative = source.relative_to(image_root)
    except ValueError:
        raise SystemExit(
            f"STOP: cached image is outside image root: {source}"
        )

    if not source.is_file() or source.stat().st_size == 0:
        raise SystemExit(
            f"STOP: cached image is missing or empty: {source}"
        )

    relative_paths.append(relative.as_posix())
    total_bytes += source.stat().st_size

if len(set(relative_paths)) != 60_885:
    raise SystemExit("STOP: cached manifest does not map to 60885 unique files")

relative_paths = sorted(relative_paths)

nul_path.write_bytes(
    b"".join(
        path.encode("utf-8") + b"\0"
        for path in relative_paths
    )
)

text_path.write_text(
    "\n".join(relative_paths) + "\n",
    encoding="utf-8",
)

plan_sha = digest(plan_path)
plan_report = json.loads(
    report_path.read_text(encoding="utf-8")
)

if plan_sha != expected_plan_sha:
    raise SystemExit(
        f"STOP: unexpected training-plan digest: {plan_sha}"
    )

if plan_report.get("sha256") != plan_sha:
    raise SystemExit(
        "STOP: training-plan report contains a different digest"
    )

audit = {
    "schema": 1,
    "cached_manifest": str(cached_path),
    "cached_manifest_sha256": digest(cached_path),
    "training_plan": str(plan_path),
    "training_plan_sha256": plan_sha,
    "rows": int(len(frame)),
    "unique_record_ids": int(frame["record_id"].nunique()),
    "unique_image_files": len(relative_paths),
    "image_bytes": total_bytes,
    "image_root": str(image_root),
}

audit_path.write_text(
    json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print(json.dumps(audit, indent=2, ensure_ascii=False))
PY

echo
echo "=== COPY SELECTED INATURALIST IMAGES ==="

mkdir -p "$STAGING/image_cache/inaturalist"

rsync \
    -a \
    --from0 \
    --files-from="$FILE_LIST_NUL" \
    --info=stats2 \
    "$IMAGE_ROOT/" \
    "$STAGING/image_cache/inaturalist/"

echo
echo "=== COPY MANIFESTS AND CORRECTED PLAN ==="

mkdir -p \
    "$STAGING/pipeline/image_download" \
    "$STAGING/pipeline/planning" \
    "$STAGING/job_metadata" \
    "$STAGING/configs" \
    "$STAGING/slurm"

rsync \
    -a \
    "$PLAN_DIR/" \
    "$STAGING/pipeline/planning/training_plan_v1/"

for file in \
    inaturalist_plan.parquet \
    inaturalist_cached.parquet \
    inaturalist_replacement_candidate.json \
    inaturalist_replacement_applied.json
do
    cp \
        "$DOWNLOAD_DIR/$file" \
        "$STAGING/pipeline/image_download/$file"
done

for file in \
    inat_image_job.env \
    replacement_job.env \
    replacement_audit_job.env \
    inat_probe_job.env
do
    if [[ -f "$DATA_ROOT/pipeline/$file" ]]
    then
        cp \
            "$DATA_ROOT/pipeline/$file" \
            "$STAGING/job_metadata/$file"
    fi
done

cp "$FILE_LIST_TXT" "$STAGING/IMAGE_FILES.txt"
cp "$INPUT_AUDIT" "$STAGING/INPUT_AUDIT.json"

cp \
    configs/hymenoptera_teacher_300k_v1.json \
    "$STAGING/configs/"

cp -R \
    configs/hymenoptera_teacher_plan_v1 \
    "$STAGING/configs/"

cp \
    slurm/31_hymenoptera_training_plan.sbatch \
    slurm/34_hymenoptera_inaturalist_images.sbatch \
    slurm/35_hymenoptera_inaturalist_replacement.sbatch \
    "$STAGING/slurm/"

echo
echo "=== WRITE PROVENANCE ==="

source "$DATA_ROOT/pipeline/inat_image_job.env"
source "$DATA_ROOT/pipeline/replacement_job.env"

cat > "$STAGING/PROVENANCE.txt" <<EOF
snapshot_id=hymenoptera_inaturalist_images_v1
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
git_commit=$(git rev-parse HEAD)
git_branch=$(git branch --show-current)
training_plan_rows=300000
training_plan_sha256=$EXPECTED_PLAN_SHA
inaturalist_rows=60885
replacement_job=$replacement_job
image_download_job=$inat_image_job
failed_record_id=img:280b341e6dad08f6e4d48a29
replacement_record_id=img:89af631c166b5a75b4de01ed
contents=selected_images,cached_manifest,corrected_plan,replacement_reports,configs,slurm
EOF

echo
echo "=== VERIFY COPIED IMAGE SET ==="

copied_files=$(
    find "$STAGING/image_cache/inaturalist" \
        -type f |
    wc -l
)

if [[ "$copied_files" -ne 60885 ]]
then
    echo "STOP: copied $copied_files images instead of 60885"
    exit 1
fi

if find "$STAGING/image_cache/inaturalist" \
    -type f \
    -empty \
    -print -quit |
    grep -q .
then
    echo "STOP: permanent copy contains an empty image"
    exit 1
fi

echo "Copied and validated images: $copied_files"

echo
echo "=== CREATE AND VERIFY CHECKSUMS ==="

(
    cd "$STAGING"

    find . \
        -type f \
        ! -name SHA256SUMS.txt \
        -print0 |
    LC_ALL=C sort -z |
    xargs -0 sha256sum \
        > SHA256SUMS.txt

    sha256sum --quiet -c SHA256SUMS.txt
)

echo "All checksums verified"

echo
echo "=== FINALIZE ATOMICALLY ==="

mv "$STAGING" "$BACKUP"

echo
echo "=== PERMANENT CHECKPOINT ==="

du -sh "$BACKUP"
wc -l "$BACKUP/SHA256SUMS.txt"
cat "$BACKUP/PROVENANCE.txt"

echo
echo "Hymenoptera iNaturalist checkpoint completed"
echo "Scratch files were not deleted"
