#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-train
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "$REPO_DIR"
echo "Compute node: $(hostname)"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/run_multisource_scale.py \
    --stage train \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --source-root "$SOURCE_ROOT" \
    --image-root "$IMAGE_ROOT"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/run_multisource_scale.py \
    --stage evaluate \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --source-root "$SOURCE_ROOT" \
    --image-root "$IMAGE_ROOT"
