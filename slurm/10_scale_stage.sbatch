#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-scale-stage
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
CONFIG="${CONFIG:-configs/multisource_scale_v10_raw200k.json}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v10_raw200k}"
SOURCE_ROOT="${SOURCE_ROOT:-/scratch/$USER/taxalens/source_cache}"
IMAGE_ROOT="${IMAGE_ROOT:-/scratch/$USER/taxalens/image_cache}"
STAGE="${STAGE:-doctor}"
SOURCE="${SOURCE:-all}"

case "$STAGE" in
    doctor|checkpoint|prefetch|bioscan|bioscan-topup|download|ingest|assemble|plan|cache|freeze-eval|evaluate|keys) ;;
    *) echo "Unsupported CPU stage: $STAGE"; exit 2 ;;
esac

if [[ ! -s "$CONTAINER" ]]; then
    echo "Container not found: $CONTAINER"
    exit 1
fi
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "Python environment not found: $VENV"
    exit 1
fi

CACHE_ROOT="/scratch/$USER/taxalens/cache"
mkdir -p "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

args=(
    scripts/run_multisource_scale.py
    --stage "$STAGE"
    --config "$CONFIG"
    --data-root "$DATA_ROOT"
    --source-root "$SOURCE_ROOT"
    --image-root "$IMAGE_ROOT"
)

if [[ "$STAGE" == "download" ]]; then
    args+=(--source "$SOURCE")
fi
if [[ "${REUSE_EXISTING_BIOSCAN:-0}" == "1" ]]; then
    args+=(--reuse-existing-bioscan)
fi
if [[ "${ALLOW_INAT_BULK_DOWNLOAD:-0}" == "1" ]]; then
    args+=(--allow-inat-bulk-download)
fi
if [[ -n "${INAT_ARCHIVE:-}" ]]; then
    args+=(--inat-archive "$INAT_ARCHIVE")
fi
if [[ "${SUBMIT_GBIF_REQUEST:-0}" == "1" ]]; then
    args+=(--submit-gbif-request)
fi
if [[ -n "${GBIF_DOWNLOAD_KEY:-}" ]]; then
    args+=(--gbif-download-key "$GBIF_DOWNLOAD_KEY")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    args+=(--dry-run)
fi
if [[ "$STAGE" == "keys" ]]; then
    args+=(--family "${FAMILY:-}" --genera "${GENERA:-}")
fi

echo "Compute node: $(hostname)"
echo "Stage: $STAGE"
echo "Config: $CONFIG"
echo "Data root: $DATA_ROOT"
echo "Shared source root: $SOURCE_ROOT"
echo "Shared image cache: $IMAGE_ROOT"

apptainer exec "$CONTAINER" "$VENV/bin/python" "${args[@]}"
