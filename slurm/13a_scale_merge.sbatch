#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-merge-models
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v10_raw200k_swiss28}"

PIPELINE="$DATA_ROOT/pipeline"
SHARDS="$PIPELINE/quality_manifest_shards"
EMBED_ROOT="$PIPELINE/quality_embedding_shards"
MODELS_ROOT="$DATA_ROOT/models"
DINO_DIR="$MODELS_ROOT/comparisons/dino"
BIOCLIP_DIR="$MODELS_ROOT/comparisons/bioclip"

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"
cd "$REPO_DIR"

for encoder in dino bioclip; do
    count="$(find "$EMBED_ROOT/$encoder" -type f -name complete.json | wc -l)"
    echo "$encoder completed shards: $count/159"
    [[ "$count" -eq 159 ]] || {
        echo "STOP: expected 159 completed $encoder shards"
        exit 1
    }
done

mkdir -p "$DINO_DIR" "$BIOCLIP_DIR"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/merge_embedding_shards.py \
    --shard-root "$EMBED_ROOT/dino" \
    --manifest-root "$SHARDS" \
    --out-dir "$DINO_DIR"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/merge_embedding_shards.py \
    --shard-root "$EMBED_ROOT/bioclip" \
    --manifest-root "$SHARDS" \
    --out-dir "$BIOCLIP_DIR"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/fuse_embedding_models.py \
    --dino-dir "$DINO_DIR" \
    --bioclip-dir "$BIOCLIP_DIR" \
    --out-dir "$MODELS_ROOT"

echo "Merge and fusion completed"
