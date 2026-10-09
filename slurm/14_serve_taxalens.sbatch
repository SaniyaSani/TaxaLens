#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-serve
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
MODEL_DIR="${MODEL_DIR:-/scratch/$USER/taxalens/v10_raw200k_swiss28/models}"
PORT="${PORT:-8765}"

CACHE_ROOT="/scratch/$USER/taxalens/cache"
mkdir -p "$CACHE_ROOT/huggingface" "$CACHE_ROOT/torch"

export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "TaxaLens URL inside cluster: http://$(hostname):$PORT"
echo "Model directory: $MODEL_DIR"

nvidia-smi

apptainer exec --nv "$CONTAINER" \
  "$VENV/bin/python" \
  scripts/serve_taxalens.py \
  --model-dir "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port "$PORT"
