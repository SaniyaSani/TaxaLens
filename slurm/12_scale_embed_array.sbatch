#!/usr/bin/bash -l
# Submit with: sbatch --array=0-N%10 --export=ALL,CONFIG=...,DATA_ROOT=... slurm/12_scale_embed_array.sbatch
#SBATCH --job-name=taxalens-embed
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=lowprio
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=L4:1
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch --array=0-N%CONCURRENCY}"

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
CONFIG="${CONFIG:-configs/multisource_scale_v10_raw200k.json}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v10_raw200k}"
SOURCE_ROOT="${SOURCE_ROOT:-/scratch/$USER/taxalens/source_cache}"
IMAGE_ROOT="${IMAGE_ROOT:-/scratch/$USER/taxalens/image_cache}"
ENCODER="${ENCODER:-all}"

case "$ENCODER" in
    all|dino|bioclip) ;;
    *) echo "ENCODER must be all, dino or bioclip"; exit 2 ;;
esac

if [[ ! -s "$CONTAINER" || ! -x "$VENV/bin/python" ]]; then
    echo "TaxaLens container or Python environment is missing"
    exit 1
fi

CACHE_ROOT="/scratch/$USER/taxalens/cache"
mkdir -p "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch"
export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "$REPO_DIR"
echo "Compute node: $(hostname)"
echo "Embedding shard index: $SLURM_ARRAY_TASK_ID"
nvidia-smi

apptainer exec --nv "$CONTAINER" "$VENV/bin/python" \
    scripts/run_multisource_scale.py \
    --stage embed \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --source-root "$SOURCE_ROOT" \
    --image-root "$IMAGE_ROOT" \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --encoder "$ENCODER"
