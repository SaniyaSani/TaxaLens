#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-report
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"

MASTER="$DATA_ROOT/pipeline/corpus/master_manifest.parquet"
REPORT_JSON="$DATA_ROOT/pipeline/master_corpus_report.json"
REPORT_MD="$DATA_ROOT/pipeline/master_corpus_report.md"

if [[ ! -s "$MASTER" ]]
then
    echo "STOP: master manifest is missing or empty: $MASTER"
    exit 1
fi

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Master manifest: $MASTER"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/corpus_report.py \
    --input "$MASTER" \
    --require-sources "iNaturalist,BIOSCAN-5M,GBIF" \
    --out-json "$REPORT_JSON" \
    --out-md "$REPORT_MD"

echo
echo "=== MASTER CORPUS REPORT ==="

sed -n '1,260p' "$REPORT_MD"

echo
echo "=== REPORT FILES ==="

ls -lh "$REPORT_JSON" "$REPORT_MD"

echo
echo "Hymenoptera master corpus report completed"
