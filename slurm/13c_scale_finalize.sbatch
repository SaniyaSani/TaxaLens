#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-finalize
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v10_raw200k_swiss28}"
MODELS_ROOT="$DATA_ROOT/models"

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"
cd "$REPO_DIR"

for model_dir in \
    "$MODELS_ROOT/comparisons/dino" \
    "$MODELS_ROOT/comparisons/bioclip" \
    "$MODELS_ROOT"
do
    test -s "$model_dir/classifiers.joblib"
    test -s "$model_dir/training_report_hierarchical.json"
done

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/build_retrieval_index.py \
    --model-dir "$MODELS_ROOT"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/render_evaluation_v09.py \
    --report "$MODELS_ROOT/training_report_hierarchical.json" \
    --out "$MODELS_ROOT/evaluation_by_source.md"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/compare_encoder_reports.py \
    --dino "$MODELS_ROOT/comparisons/dino/training_report_hierarchical.json" \
    --bioclip "$MODELS_ROOT/comparisons/bioclip/training_report_hierarchical.json" \
    --fusion "$MODELS_ROOT/training_report_hierarchical.json" \
    --out-json "$MODELS_ROOT/encoder_comparison.json" \
    --out-md "$MODELS_ROOT/encoder_comparison.md"

echo "Retrieval index and evaluation reports completed"
