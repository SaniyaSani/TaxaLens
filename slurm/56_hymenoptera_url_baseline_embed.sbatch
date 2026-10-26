#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-url-embed
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=lowprio
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=L4:1
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?This job must be submitted as an array}"

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"

BASELINE="$DATA_ROOT/pipeline/url_baseline_v1"
SHARDS="$BASELINE/manifest_shards"
EMBED_ROOT="$BASELINE/embedding_shards"

TAG=$(printf '%05d' "$SLURM_ARRAY_TASK_ID")
MANIFEST="$SHARDS/shard_${TAG}.parquet"

DINO_OUT="$EMBED_ROOT/dino/shard_${TAG}"
BIOCLIP_OUT="$EMBED_ROOT/bioclip/shard_${TAG}"

if [[ ! -s "$MANIFEST" ]]
then
    echo "STOP: embedding manifest does not exist: $MANIFEST"
    exit 1
fi

CACHE_ROOT="/scratch/$USER/taxalens/cache"

mkdir -p \
    "$CACHE_ROOT/huggingface" \
    "$CACHE_ROOT/torch" \
    "$DINO_OUT" \
    "$BIOCLIP_OUT"

export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Embedding shard: $SLURM_ARRAY_TASK_ID/145"
echo "Manifest: $MANIFEST"
nvidia-smi

echo
echo "=== DINOV3 EMBEDDINGS ==="

apptainer exec --nv "$CONTAINER" "$VENV/bin/python" \
    scripts/process_embedding_shard.py \
    --manifest "$MANIFEST" \
    --out-dir "$DINO_OUT" \
    --backend dino \
    --model facebook/dinov3-vitb16-pretrain-lvd1689m \
    --image-size 512 \
    --tile-grid 1 \
    --batch-size 4 \
    --max-image-mb 50 \
    --min-success-rate 1.0 \
    --retries 1 \
    --local-only

echo
echo "=== BIOCLIP 2 EMBEDDINGS ==="

apptainer exec --nv "$CONTAINER" "$VENV/bin/python" \
    scripts/process_embedding_shard.py \
    --manifest "$MANIFEST" \
    --out-dir "$BIOCLIP_OUT" \
    --backend bioclip \
    --model hf-hub:imageomics/bioclip-2 \
    --image-size 224 \
    --tile-grid 1 \
    --batch-size 8 \
    --max-image-mb 50 \
    --min-success-rate 1.0 \
    --retries 1 \
    --local-only

test -s "$DINO_OUT/complete.json"
test -s "$BIOCLIP_OUT/complete.json"

echo
echo "Embedding shard $TAG completed for both encoders"
