#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-assemble
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"

CORPUS_DIR="$DATA_ROOT/pipeline/corpus"

BIOSCAN="$CORPUS_DIR/bioscan_raw.parquet"
INAT="$CORPUS_DIR/inat_raw.parquet"
GBIF="$CORPUS_DIR/gbif_raw.parquet"

COMBINED="$CORPUS_DIR/combined_before_dedup.parquet"
DEDUPLICATED="$CORPUS_DIR/master_deduplicated.parquet"
MASTER="$CORPUS_DIR/master_manifest.parquet"

mkdir -p "$CORPUS_DIR"

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Taxonomic order: Hymenoptera"
echo "Corpus directory: $CORPUS_DIR"

for input in "$BIOSCAN" "$INAT" "$GBIF"
do
    if [[ ! -s "$input" ]]
    then
        echo "STOP: missing or empty source manifest: $input"
        exit 1
    fi
done

if [[ -s "$MASTER" ]]
then
    echo "STOP: master manifest already exists: $MASTER"
    echo "Existing corpus will not be overwritten"
    exit 1
fi

echo
echo "=== SOURCE MANIFESTS ==="

ls -lh "$BIOSCAN" "$INAT" "$GBIF"

echo
echo "=== COMBINE SOURCES ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/build_master_manifest.py \
    "$BIOSCAN" \
    "$INAT" \
    "$GBIF" \
    --out "$COMBINED"

echo
echo "=== DEDUPLICATE RECORDS ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/deduplicate_records.py \
    --input "$COMBINED" \
    --out "$DEDUPLICATED"

echo
echo "=== ASSIGN FINAL GROUP SPLITS ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/build_master_manifest.py \
    "$DEDUPLICATED" \
    --out "$MASTER"

echo
echo "=== FINAL CORPUS FILES ==="

ls -lh \
    "$COMBINED" \
    "$DEDUPLICATED" \
    "$MASTER"

echo
echo "Hymenoptera master corpus assembled successfully"
