#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-bioscan-normalize
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"

CORPUS_DIR="$DATA_ROOT/pipeline/corpus"
SELECTION="$CORPUS_DIR/bioscan_selected.csv"
OUT="$CORPUS_DIR/bioscan_raw.parquet"

if [[ ! -s "$SELECTION" ]]
then
    echo "STOP: BIOSCAN selection is missing or empty: $SELECTION"
    exit 1
fi

if [[ -s "$OUT" ]]
then
    echo "STOP: normalized BIOSCAN manifest already exists: $OUT"
    echo "Existing output will not be overwritten"
    exit 1
fi

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Order: Hymenoptera"
echo "Selection: $SELECTION"
echo "Output: $OUT"
echo "No images will be downloaded"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/ingest_bioscan.py \
    --metadata "$SELECTION" \
    --order Hymenoptera \
    --out "$OUT" \
    --chunksize 100000

echo
echo "=== NORMALIZED BIOSCAN MANIFEST ==="

ls -lh "$OUT"

echo
echo "BIOSCAN Hymenoptera normalization completed"
