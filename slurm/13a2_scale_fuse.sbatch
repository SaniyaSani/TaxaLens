#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-fuse-models
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v10_raw200k_swiss28}"

MODELS="$DATA_ROOT/models"
DINO="$MODELS/comparisons/dino"
BIOCLIP="$MODELS/comparisons/bioclip"

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"
cd "$REPO_DIR"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/fuse_embedding_models.py \
    --dino-dir "$DINO" \
    --bioclip-dir "$BIOCLIP" \
    --out-dir "$MODELS"

test -s "$MODELS/embeddings.npy"
test -s "$MODELS/embedded_manifest.csv"
test -s "$MODELS/embedding_config.json"

echo "Fusion completed successfully"
