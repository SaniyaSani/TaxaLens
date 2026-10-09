#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-train-heads
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?Submit with --array=0-2}"

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
CONFIG="${CONFIG:-configs/multisource_scale_v10_raw200k.json}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v10_raw200k_swiss28}"
MODELS_ROOT="$DATA_ROOT/models"

case "$SLURM_ARRAY_TASK_ID" in
    0)
        MODEL_NAME="dino"
        MODEL_DIR="$MODELS_ROOT/comparisons/dino"
        ;;
    1)
        MODEL_NAME="bioclip"
        MODEL_DIR="$MODELS_ROOT/comparisons/bioclip"
        ;;
    2)
        MODEL_NAME="fusion"
        MODEL_DIR="$MODELS_ROOT"
        ;;
    *)
        echo "Invalid array index: $SLURM_ARRAY_TASK_ID"
        exit 2
        ;;
esac

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$REPO_DIR"

mapfile -t training_settings < <(
    apptainer exec "$CONTAINER" "$VENV/bin/python" -c \
    'import json,sys; t=json.load(open(sys.argv[1]))["training"]; [print(t[k]) for k in ("min_family","min_genus","min_species","species_label_quality","gate_quantile","gate_margin")]' \
    "$CONFIG"
)

[[ "${#training_settings[@]}" -eq 6 ]] || {
    echo "Could not load training settings"
    exit 1
}

COHORT="$MODELS_ROOT/embedded_manifest.csv"
test -s "$MODEL_DIR/embeddings.npy"
test -s "$MODEL_DIR/embedded_manifest.csv"
test -s "$COHORT"

echo "Training target: $MODEL_NAME"
echo "Model directory: $MODEL_DIR"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/train_hierarchical.py \
    --model-dir "$MODEL_DIR" \
    --cohort-manifest "$COHORT" \
    --min-family "${training_settings[0]}" \
    --min-genus "${training_settings[1]}" \
    --min-species "${training_settings[2]}" \
    --species-label-quality "${training_settings[3]}" \
    --gate-quantile "${training_settings[4]}" \
    --gate-margin "${training_settings[5]}"

test -s "$MODEL_DIR/classifiers.joblib"
test -s "$MODEL_DIR/training_report_hierarchical.json"

echo "Training completed: $MODEL_NAME"
