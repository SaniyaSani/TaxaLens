#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-gbif-ingest
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"

GBIF_KEY="0018027-260928105237408"
ARCHIVE="$DATA_ROOT/source/gbif_hymenoptera_${GBIF_KEY}.zip"
EXTRACT_DIR="$DATA_ROOT/source/gbif_hymenoptera_${GBIF_KEY}"
OUT="$DATA_ROOT/pipeline/corpus/gbif_raw.parquet"
JOB_TMP="$DATA_ROOT/tmp/gbif-${SLURM_JOB_ID}"

mkdir -p \
    "$EXTRACT_DIR" \
    "$DATA_ROOT/pipeline/corpus" \
    "$JOB_TMP"

export TMPDIR="$JOB_TMP"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Order: Hymenoptera"
echo "GBIF archive: $ARCHIVE"
echo "Output: $OUT"
echo "Temporary directory: $TMPDIR"

if [[ ! -s "$ARCHIVE" ]]
then
    echo "STOP: GBIF archive is missing or empty: $ARCHIVE"
    exit 1
fi

if [[ ! -s "$EXTRACT_DIR/occurrence.txt" ]] || \
   [[ ! -s "$EXTRACT_DIR/multimedia.txt" ]]
then
    echo
    echo "Extracting GBIF metadata tables..."

    unzip -oq "$ARCHIVE" \
        occurrence.txt \
        multimedia.txt \
        meta.xml \
        rights.txt \
        citations.txt \
        -d "$EXTRACT_DIR"
else
    echo "Existing extracted GBIF tables found; reusing them"
fi

for required in occurrence.txt multimedia.txt meta.xml
do
    if [[ ! -s "$EXTRACT_DIR/$required" ]]
    then
        echo "STOP: required GBIF file is missing: $required"
        exit 1
    fi
done

echo
echo "Extracted GBIF files:"

ls -lh \
    "$EXTRACT_DIR/occurrence.txt" \
    "$EXTRACT_DIR/multimedia.txt" \
    "$EXTRACT_DIR/meta.xml" \
    "$EXTRACT_DIR/rights.txt" \
    "$EXTRACT_DIR/citations.txt"

echo
echo "Normalizing licensed Hymenoptera images..."

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/ingest_gbif.py \
    --order Hymenoptera \
    --occurrence "$EXTRACT_DIR/occurrence.txt" \
    --multimedia "$EXTRACT_DIR/multimedia.txt" \
    --out "$OUT" \
    --licenses CC0,CC-BY,CC-BY-SA \
    --chunksize 100000

echo
echo "GBIF Hymenoptera ingest completed"

ls -lh "$OUT"

echo "No image files were downloaded"
