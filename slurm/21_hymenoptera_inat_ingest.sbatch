#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-inat
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
SOURCE_ROOT="${SOURCE_ROOT:-/scratch/$USER/taxalens/source_cache}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"

OUT="$DATA_ROOT/pipeline/corpus/inat_raw.parquet"
JOB_TMP="$DATA_ROOT/tmp/inat-${SLURM_JOB_ID}"

mkdir -p \
    "$DATA_ROOT/pipeline/corpus" \
    "$JOB_TMP"

export TMPDIR="$JOB_TMP"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Order: Hymenoptera"
echo "Output: $OUT"
echo "Temporary SQLite directory: $TMPDIR"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/ingest_inat.py \
    --order Hymenoptera \
    --observations "$SOURCE_ROOT/inaturalist/observations.csv.gz" \
    --photos "$SOURCE_ROOT/inaturalist/photos.csv.gz" \
    --taxa "$SOURCE_ROOT/inaturalist/taxa.csv.gz" \
    --observers "$SOURCE_ROOT/inaturalist/observers.csv.gz" \
    --out "$OUT" \
    --quality-grades research \
    --licenses CC0,CC-BY,CC-BY-SA \
    --image-size medium \
    --chunksize 100000

echo "iNaturalist Hymenoptera ingest completed"
ls -lh "$OUT"
