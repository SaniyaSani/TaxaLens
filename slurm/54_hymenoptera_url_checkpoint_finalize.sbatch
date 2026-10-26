#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-url-finalize
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"
IMAGE_ROOT="${IMAGE_ROOT:-/scratch/$USER/taxalens/image_cache}"

: "${URL_COPY_ARRAY_JOB:?URL_COPY_ARRAY_JOB is required}"

PLAN_DIR="$DATA_ROOT/pipeline/planning/training_plan_gbif_recovered_v1"
CHECKPOINT_PARENT="/shares/hawlitschek.ieu.uzh/$USER/TaxaLens/v11_hymenoptera_teacher"
FINAL="$CHECKPOINT_PARENT/url_images_and_plan_v1"
STAGE="$CHECKPOINT_PARENT/.url_images_and_plan_v1.pending-${URL_COPY_ARRAY_JOB}"

cd "$REPO_DIR"

if [[ -e "$FINAL" ]]
then
    echo "STOP: permanent checkpoint already exists: $FINAL"
    exit 1
fi

echo "=== VERIFY EIGHT COPY SHARDS ==="

for task in $(seq 0 7)
do
    tag=$(printf '%02d' "$task")
    marker="$STAGE/tasks/shard_${tag}.complete"

    if [[ ! -f "$marker" ]]
    then
        echo "STOP: copy shard $tag is not complete"
        exit 1
    fi
done

image_count=$(
    find "$STAGE/images" -type f | wc -l
)

if [[ "$image_count" -ne 144817 ]]
then
    echo "STOP: expected 144817 images, found $image_count"
    exit 1
fi

echo "Copied image files: $image_count"

echo
echo "=== COPY PLAN AND AUDIT REPORTS ==="

mkdir -p \
    "$STAGE/plan" \
    "$STAGE/reports/image_download" \
    "$STAGE/code" \
    "$STAGE/provenance"

rsync --archive \
    "$PLAN_DIR/" \
    "$STAGE/plan/"

rsync \
    --archive \
    --prune-empty-dirs \
    --include='*/' \
    --include='*.json' \
    --exclude='*' \
    "$DATA_ROOT/pipeline/image_download/" \
    "$STAGE/reports/image_download/"

code_paths=(
    scripts/download_images.py
    tests/test_download_images_retry.py
    configs/hymenoptera_teacher_300k_v1.json
    configs/hymenoptera_teacher_plan_v1
)

while IFS= read -r file
do
    code_paths+=("$file")
done < <(
    find slurm \
        -maxdepth 1 \
        -type f \
        -name '*.sbatch' \
        -printf '%p\n' |
    sort
)

rsync --archive --relative \
    "${code_paths[@]}" \
    "$STAGE/code/"

echo
echo "=== CREATE PORTABLE TRAINING PLAN ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - "$STAGE/plan/training_plan.parquet" \
    "$STAGE/plan/training_plan_portable.parquet" \
    "$IMAGE_ROOT" \
    "$STAGE/provenance/checkpoint_report.json" \
    "$URL_COPY_ARRAY_JOB" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

source_path = Path(sys.argv[1])
portable_path = Path(sys.argv[2])
image_root = Path(sys.argv[3]).resolve()
report_path = Path(sys.argv[4])
copy_job = sys.argv[5]

frame = pd.read_parquet(source_path)
url_mask = frame["source"].astype(str).isin(
    ["GBIF", "iNaturalist"]
)

portable_paths = []

for value in frame.loc[url_mask, "local_path"].astype(str):
    local = Path(value).resolve()

    try:
        relative = local.relative_to(image_root)
    except ValueError as exc:
        raise SystemExit(
            f"STOP: local path outside image cache: {local}"
        ) from exc

    portable_paths.append(
        "images/" + relative.as_posix()
    )

frame.loc[url_mask, "local_path"] = portable_paths
frame.loc[
    frame["source"].astype(str).eq("BIOSCAN-5M"),
    "local_path",
] = ""

pending = portable_path.with_suffix(".parquet.pending")
frame.to_parquet(pending, index=False)
pending.replace(portable_path)

source_counts = {
    str(key): int(value)
    for key, value in frame["source"].value_counts().items()
}

if len(frame) != 298_932:
    raise SystemExit(
        f"STOP: expected 298,932 rows, found {len(frame):,}"
    )

if source_counts != {
    "BIOSCAN-5M": 154115,
    "GBIF": 81457,
    "iNaturalist": 63360,
}:
    raise SystemExit(
        f"STOP: unexpected source counts: {source_counts}"
    )

def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()

try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"],
        text=True,
    ).strip()
except Exception:
    commit = "unknown"
    branch = "unknown"

report = {
    "schema": 1,
    "snapshot_id": "hymenoptera_url_images_and_plan_v1",
    "rows": len(frame),
    "source_counts": source_counts,
    "url_images": int(url_mask.sum()),
    "bioscan_images_pending": int((~url_mask).sum()),
    "training_plan_sha256": digest(source_path),
    "portable_plan_sha256": digest(portable_path),
    "copy_array_job": copy_job,
    "git_commit": commit,
    "git_branch": branch,
    "contents": [
        "selected_gbif_images",
        "selected_inaturalist_images",
        "final_training_plan",
        "portable_training_plan",
        "replacement_ledger",
        "dropped_targets",
        "audit_reports",
        "code_snapshot",
    ],
}

pending_report = report_path.with_suffix(".json.pending")
pending_report.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
pending_report.replace(report_path)

print(json.dumps(report, indent=2, ensure_ascii=False))
PY

echo
echo "=== CREATE COMPLETE CHECKSUM MANIFEST ==="

(
    cd "$STAGE"

    cat tasks/shard_*.sha256 \
        > SHA256SUMS.txt.pending

    find \
        plan \
        reports \
        code \
        provenance \
        tasks \
        -type f \
        -print0 |
    sort -z |
    xargs -0 sha256sum \
        >> SHA256SUMS.txt.pending

    mv SHA256SUMS.txt.pending SHA256SUMS.txt
)

checksum_rows=$(
    wc -l < "$STAGE/SHA256SUMS.txt"
)

echo "Checksum entries: $checksum_rows"

echo
echo "=== VERIFY COMPLETE CHECKPOINT ==="

(
    cd "$STAGE"
    sha256sum --quiet -c SHA256SUMS.txt
)

echo "All permanent-checkpoint checksums match"

echo
echo "=== FINALIZE ATOMICALLY ==="

mv "$STAGE" "$FINAL"

echo
echo "=== PERMANENT CHECKPOINT ==="

du -sh "$FINAL"
find "$FINAL/images" -type f | wc -l
ls -lh \
    "$FINAL/plan/training_plan.parquet" \
    "$FINAL/plan/training_plan_portable.parquet" \
    "$FINAL/SHA256SUMS.txt"

python -m json.tool \
    "$FINAL/provenance/checkpoint_report.json"

echo
echo "Hymenoptera URL-image checkpoint completed"
echo "Scratch files were not deleted"
