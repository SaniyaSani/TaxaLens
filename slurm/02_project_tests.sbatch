#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-tests
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif"
VENV="$HOME/.venvs/taxalens-v09"

CACHE_ROOT="/scratch/$USER/taxalens/cache"

mkdir -p \
    "$CACHE_ROOT/huggingface" \
    "$CACHE_ROOT/torch"

export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Checking Python files..."

apptainer exec "$CONTAINER" \
    "$VENV/bin/python" -m compileall -q src scripts

echo "Running TaxaLens tests..."

apptainer exec "$CONTAINER" \
    "$VENV/bin/python" -m pytest -q

echo "All TaxaLens tests passed"
