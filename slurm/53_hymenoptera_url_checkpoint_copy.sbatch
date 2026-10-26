#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-url-copy
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-7%8
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"
IMAGE_ROOT="${IMAGE_ROOT:-/scratch/$USER/taxalens/image_cache}"

PLAN="$DATA_ROOT/pipeline/planning/training_plan_gbif_recovered_v1/training_plan.parquet"

CHECKPOINT_PARENT="/shares/hawlitschek.ieu.uzh/$USER/TaxaLens/v11_hymenoptera_teacher"
FINAL="$CHECKPOINT_PARENT/url_images_and_plan_v1"
STAGE="$CHECKPOINT_PARENT/.url_images_and_plan_v1.pending-${SLURM_ARRAY_JOB_ID}"

SHARDS=8
TASK_ID="$SLURM_ARRAY_TASK_ID"
TAG=$(printf '%02d' "$TASK_ID")

WORK="$DATA_ROOT/tmp/url-checkpoint-${SLURM_ARRAY_JOB_ID}"
FILE_LIST="$WORK/shard_${TAG}.files"
TASK_SHA="$STAGE/tasks/shard_${TAG}.sha256"
TASK_REPORT="$STAGE/tasks/shard_${TAG}.json"
TASK_MARKER="$STAGE/tasks/shard_${TAG}.complete"

mkdir -p \
    "$WORK" \
    "$STAGE/images" \
    "$STAGE/tasks"

if [[ -e "$FINAL" ]]
then
    echo "STOP: permanent checkpoint already exists: $FINAL"
    exit 1
fi

cd "$REPO_DIR"

echo "Array job: ${SLURM_ARRAY_JOB_ID}"
echo "Shard: $TASK_ID/$SHARDS"
echo "Plan: $PLAN"
echo "Destination: $STAGE"
echo

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - "$PLAN" "$IMAGE_ROOT" "$TASK_ID" "$SHARDS" \
    "$FILE_LIST" "$TASK_SHA" "$TASK_REPORT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

plan_path = Path(sys.argv[1])
image_root = Path(sys.argv[2]).resolve()
task_id = int(sys.argv[3])
shards = int(sys.argv[4])
list_path = Path(sys.argv[5])
sha_path = Path(sys.argv[6])
report_path = Path(sys.argv[7])

frame = pd.read_parquet(
    plan_path,
    columns=[
        "record_id",
        "source",
        "local_path",
        "image_sha256",
    ],
)

selected = frame[
    frame["source"].fillna("").astype(str).isin(
        ["GBIF", "iNaturalist"]
    )
].copy()

if len(selected) != 144_817:
    raise SystemExit(
        f"STOP: expected 144,817 URL-image rows, found {len(selected):,}"
    )

if selected["record_id"].astype(str).duplicated().any():
    raise SystemExit("STOP: duplicate URL-image record IDs")

if selected["local_path"].fillna("").astype(str).eq("").any():
    raise SystemExit("STOP: empty local_path in URL-image rows")

if selected["local_path"].astype(str).duplicated().any():
    raise SystemExit("STOP: duplicate local image paths")

entries = []
source_counts = {}

for row in selected.itertuples(index=False):
    record_id = str(row.record_id)

    shard = int.from_bytes(
        hashlib.sha256(record_id.encode("utf-8")).digest()[:8],
        "big",
    ) % shards

    if shard != task_id:
        continue

    source = str(row.source)
    local = Path(str(row.local_path)).resolve()
    sha = str(row.image_sha256).strip().lower()

    try:
        relative = local.relative_to(image_root)
    except ValueError as exc:
        raise SystemExit(
            f"STOP: image outside cache root: {local}"
        ) from exc

    if not local.is_file() or local.stat().st_size == 0:
        raise SystemExit(f"STOP: missing source image: {local}")

    if len(sha) != 64 or any(
        character not in "0123456789abcdef" for character in sha
    ):
        raise SystemExit(
            f"STOP: invalid image SHA-256 for {record_id}: {sha!r}"
        )

    if relative.parts[0] not in {"gbif", "inaturalist"}:
        raise SystemExit(
            f"STOP: unexpected cache directory: {relative}"
        )

    entries.append((relative.as_posix(), sha))
    source_counts[source] = source_counts.get(source, 0) + 1

entries.sort()

list_path.parent.mkdir(parents=True, exist_ok=True)
sha_path.parent.mkdir(parents=True, exist_ok=True)
report_path.parent.mkdir(parents=True, exist_ok=True)

list_pending = list_path.with_suffix(".files.pending")
sha_pending = sha_path.with_suffix(".sha256.pending")
report_pending = report_path.with_suffix(".json.pending")

list_pending.write_text(
    "".join(f"{relative}\n" for relative, _ in entries),
    encoding="utf-8",
)

sha_pending.write_text(
    "".join(
        f"{sha}  images/{relative}\n"
        for relative, sha in entries
    ),
    encoding="utf-8",
)

report = {
    "schema": 1,
    "array_job": int(sys.argv[3]),
    "shard": task_id,
    "shards": shards,
    "rows": len(entries),
    "source_counts": source_counts,
}

report_pending.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

list_pending.replace(list_path)
sha_pending.replace(sha_path)
report_pending.replace(report_path)

print(json.dumps(report, indent=2, ensure_ascii=False))
PY

echo
echo "=== PARALLEL SHARD COPY ==="

rsync \
    --archive \
    --files-from="$FILE_LIST" \
    --human-readable \
    --stats \
    "$IMAGE_ROOT/" \
    "$STAGE/images/"

echo
echo "=== VERIFY SHARD CHECKSUMS ==="

(
    cd "$STAGE"
    sha256sum --quiet -c "tasks/shard_${TAG}.sha256"
)

touch "$TASK_MARKER"

echo "Shard $TAG copied and verified successfully"
