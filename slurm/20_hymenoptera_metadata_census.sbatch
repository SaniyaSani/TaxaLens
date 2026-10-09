#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-census
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
SOURCE_ROOT="${SOURCE_ROOT:-/scratch/$USER/taxalens/source_cache}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

mkdir -p "$DATA_ROOT/pipeline/census"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Order: Hymenoptera"
echo "Output root: $DATA_ROOT"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/census_order_metadata.py \
    --order Hymenoptera \
    --inat-taxa "$SOURCE_ROOT/inaturalist/taxa.csv.gz" \
    --bioscan-metadata \
      "$SOURCE_ROOT/bioscan/bioscan5m/metadata/csv/BIOSCAN_5M_Insect_Dataset_metadata.csv" \
    --out "$DATA_ROOT/pipeline/census/hymenoptera_metadata_census.json" \
    --chunksize 200000
