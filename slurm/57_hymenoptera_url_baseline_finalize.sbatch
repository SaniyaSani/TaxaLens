#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-url-train
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"

BASELINE="$DATA_ROOT/pipeline/url_baseline_v1"
SHARDS="$BASELINE/manifest_shards"
EMBED_ROOT="$BASELINE/embedding_shards"
MODELS="$DATA_ROOT/models/hymenoptera_url_baseline_v1"

DINO="$MODELS/comparisons/dino"
BIOCLIP="$MODELS/comparisons/bioclip"
FUSION="$MODELS"

CHECKPOINT_PARENT="/shares/hawlitschek.ieu.uzh/$USER/TaxaLens/v11_hymenoptera_teacher"
FINAL_CHECKPOINT="$CHECKPOINT_PARENT/hymenoptera_url_baseline_v1"
STAGE_CHECKPOINT="$CHECKPOINT_PARENT/.hymenoptera_url_baseline_v1.pending-${SLURM_JOB_ID}"

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "=== VERIFY 145 COMPLETED SHARDS ==="

for encoder in dino bioclip
do
    count=$(
        find "$EMBED_ROOT/$encoder" \
            -mindepth 2 \
            -maxdepth 2 \
            -type f \
            -name complete.json |
        wc -l
    )

    echo "$encoder completed shards: $count/145"

    if [[ "$count" -ne 145 ]]
    then
        echo "STOP: incomplete $encoder embedding set"
        exit 1
    fi
done

mkdir -p "$DINO" "$BIOCLIP"

echo
echo "=== MERGE BOTH ENCODERS ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/merge_embedding_shards.py \
    --shard-root "$EMBED_ROOT/dino" \
    --manifest-root "$SHARDS" \
    --out-dir "$DINO"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/merge_embedding_shards.py \
    --shard-root "$EMBED_ROOT/bioclip" \
    --manifest-root "$SHARDS" \
    --out-dir "$BIOCLIP"

echo
echo "=== FUSE DINOV3 AND BIOCLIP ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/fuse_embedding_models.py \
    --dino-dir "$DINO" \
    --bioclip-dir "$BIOCLIP" \
    --out-dir "$FUSION"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - "$DINO" "$BIOCLIP" "$FUSION" <<'PY'
import sys
from pathlib import Path

import pandas as pd

directories = [Path(value) for value in sys.argv[1:]]
record_sets = []

for directory in directories:
    frame = pd.read_csv(
        directory / "embedded_manifest.csv",
        dtype=str,
        keep_default_na=False,
    )

    if len(frame) != 99_928:
        raise SystemExit(
            f"STOP: {directory} has {len(frame):,} rows"
        )

    if frame["record_id"].duplicated().any():
        raise SystemExit(f"STOP: duplicate record IDs in {directory}")

    record_sets.append(set(frame["record_id"]))

if not all(values == record_sets[0] for values in record_sets[1:]):
    raise SystemExit("STOP: encoder specimen cohorts differ")

print("All three embedding cohorts contain the same 99,928 specimens")
PY

echo
echo "=== TRAIN THREE HIERARCHICAL MODELS IN PARALLEL ==="

train_model() {
    local name="$1"
    local directory="$2"
    local logfile="$MODELS/train_${name}.log"

    (
        export OMP_NUM_THREADS=8
        export OPENBLAS_NUM_THREADS=8
        export MKL_NUM_THREADS=8
        export NUMEXPR_NUM_THREADS=8

        apptainer exec "$CONTAINER" "$VENV/bin/python" \
            scripts/train_hierarchical.py \
            --model-dir "$directory" \
            --cohort-manifest "$FUSION/embedded_manifest.csv" \
            --min-family 50 \
            --min-genus 25 \
            --min-species 12 \
            --species-label-quality A,B \
            --gate-quantile 0.05 \
            --gate-margin 0.02
    ) > "$logfile" 2>&1
}

train_model dino "$DINO" &
pid_dino=$!

train_model bioclip "$BIOCLIP" &
pid_bioclip=$!

train_model fusion "$FUSION" &
pid_fusion=$!

training_status=0

wait "$pid_dino" || training_status=1
wait "$pid_bioclip" || training_status=1
wait "$pid_fusion" || training_status=1

for logfile in "$MODELS"/train_*.log
do
    echo
    echo "--- $(basename "$logfile") ---"
    tail -n 30 "$logfile"
done

if [[ "$training_status" -ne 0 ]]
then
    echo "STOP: at least one hierarchical model failed"
    exit 1
fi

echo
echo "=== BUILD RETRIEVAL INDEX AND EVALUATION ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/build_retrieval_index.py \
    --model-dir "$FUSION"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/render_evaluation_v09.py \
    --report "$FUSION/training_report_hierarchical.json" \
    --out "$FUSION/evaluation_by_source.md"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/compare_encoder_reports.py \
    --dino "$DINO/training_report_hierarchical.json" \
    --bioclip "$BIOCLIP/training_report_hierarchical.json" \
    --fusion "$FUSION/training_report_hierarchical.json" \
    --out-json "$FUSION/encoder_comparison.json" \
    --out-md "$FUSION/encoder_comparison.md"

echo
echo "=== CREATE PERMANENT MODEL CHECKPOINT ==="

if [[ -e "$FINAL_CHECKPOINT" ]]
then
    echo "STOP: permanent model checkpoint already exists"
    exit 1
fi

mkdir -p \
    "$STAGE_CHECKPOINT/models" \
    "$STAGE_CHECKPOINT/pipeline" \
    "$STAGE_CHECKPOINT/config" \
    "$STAGE_CHECKPOINT/code"

rsync --archive \
    "$MODELS/" \
    "$STAGE_CHECKPOINT/models/"

rsync --archive \
    "$BASELINE/training_manifest.parquet" \
    "$BASELINE/preparation_report.json" \
    "$BASELINE/manifest_shards/shards.json" \
    "$STAGE_CHECKPOINT/pipeline/"

rsync --archive \
    configs/hymenoptera_url_baseline_v1.json \
    "$STAGE_CHECKPOINT/config/"

rsync --archive \
    scripts/make_embedding_shards.py \
    scripts/process_embedding_shard.py \
    scripts/merge_embedding_shards.py \
    scripts/fuse_embedding_models.py \
    scripts/train_hierarchical.py \
    scripts/build_retrieval_index.py \
    scripts/render_evaluation_v09.py \
    scripts/compare_encoder_reports.py \
    slurm/55_hymenoptera_url_baseline_prepare.sbatch \
    slurm/56_hymenoptera_url_baseline_embed.sbatch \
    slurm/57_hymenoptera_url_baseline_finalize.sbatch \
    "$STAGE_CHECKPOINT/code/"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - "$STAGE_CHECKPOINT" "$SLURM_JOB_ID" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

report_path = root / "models" / "training_report_hierarchical.json"
comparison_path = root / "models" / "encoder_comparison.json"
manifest_path = root / "pipeline" / "training_manifest.parquet"

report = json.loads(report_path.read_text(encoding="utf-8"))
comparison = json.loads(comparison_path.read_text(encoding="utf-8"))

provenance = {
    "schema": 1,
    "snapshot_id": "hymenoptera_url_baseline_v1",
    "training_rows": 99928,
    "encoders": ["DINOv3-ViT-B/16", "BioCLIP-2"],
    "fusion": "equal_weight_l2_concat",
    "training_manifest_sha256": digest(manifest_path),
    "classifiers_sha256": digest(
        root / "models" / "classifiers.joblib"
    ),
    "finalizer_job": sys.argv[2],
    "evaluation_context": report.get("evaluation_context", {}),
    "encoder_comparison": comparison,
}

path = root / "checkpoint_report.json"
path.write_text(
    json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY

(
    cd "$STAGE_CHECKPOINT"

    find . \
        -type f \
        ! -name SHA256SUMS.txt \
        -print0 |
    sort -z |
    xargs -0 sha256sum \
        > SHA256SUMS.txt

    sha256sum --quiet -c SHA256SUMS.txt
)

mv "$STAGE_CHECKPOINT" "$FINAL_CHECKPOINT"

echo
echo "=== FINAL BASELINE MODEL ==="

du -sh "$FINAL_CHECKPOINT"

python -m json.tool \
    "$FINAL_CHECKPOINT/checkpoint_report.json"

echo
echo "Hymenoptera URL baseline training completed"
echo "Model checkpoint is permanent on shares"
