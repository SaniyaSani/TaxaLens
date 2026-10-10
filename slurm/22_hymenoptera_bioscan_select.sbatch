#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-bioscan-select
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
SOURCE_ROOT="${SOURCE_ROOT:-/scratch/$USER/taxalens/source_cache}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"

METADATA="$SOURCE_ROOT/bioscan/bioscan5m/metadata/csv/BIOSCAN_5M_Insect_Dataset_metadata.csv"
SCOPE_DIR="$REPO_DIR/configs/hymenoptera_bioscan_scopes"
SELECT_DIR="$DATA_ROOT/pipeline/bioscan_selection"
CORPUS_DIR="$DATA_ROOT/pipeline/corpus"
IMAGE_DIR="$DATA_ROOT/images/bioscan"
COMBINED="$CORPUS_DIR/bioscan_selected.csv"

mkdir -p \
    "$SELECT_DIR" \
    "$CORPUS_DIR" \
    "$IMAGE_DIR"

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Metadata: $METADATA"
echo "No images will be downloaded during this job"

manifests=()

run_scope() {
    local name="$1"
    local target="$2"
    local split_targets="$3"

    local manifest="$SELECT_DIR/${name}.csv"
    local report="$SELECT_DIR/${name}_report.json"
    local scope="$SCOPE_DIR/${name}.json"

    echo
    echo "=== Selecting $name: target $target ==="

    apptainer exec "$CONTAINER" "$VENV/bin/python" \
        scripts/select_bioscan_diptera.py \
        --metadata "$METADATA" \
        --order Hymenoptera \
        --out "$manifest" \
        --report "$report" \
        --image-dir "$IMAGE_DIR" \
        --max-records "$target" \
        --max-per-taxon 4 \
        --min-rank family \
        --families-file "$scope" \
        --split-targets "$split_targets" \
        --seed 42 \
        --chunksize 100000 \
        --allow-shortfall

    manifests+=("$manifest")
}

run_scope \
    minute_parasitoid_enriched \
    135000 \
    "pretrain=54000,train=54000,val=6750,test=6750,key_unseen=3375,val_unseen=3375,test_unseen=3375,other_heldout=3375"

run_scope \
    braconid_ichneumonid \
    35000 \
    "pretrain=14000,train=14000,val=1750,test=1750,key_unseen=875,val_unseen=875,test_unseen=875,other_heldout=875"

run_scope \
    other_parasitoid_enriched \
    7000 \
    "pretrain=2800,train=2800,val=350,test=350,key_unseen=175,val_unseen=175,test_unseen=175,other_heldout=175"

run_scope \
    aculeata_context \
    20000 \
    "pretrain=8000,train=8000,val=1000,test=1000,key_unseen=500,val_unseen=500,test_unseen=500,other_heldout=500"

run_scope \
    mixed_and_symphyta_context \
    3000 \
    "pretrain=1200,train=1200,val=150,test=150,key_unseen=75,val_unseen=75,test_unseen=75,other_heldout=75"

echo
echo "=== Merging selection manifests ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/merge_bioscan_manifests.py \
    --out "$COMBINED" \
    "${manifests[@]}"

echo
echo "=== Selection files ==="

ls -lh "$SELECT_DIR"/*.csv "$SELECT_DIR"/*.json "$COMBINED"

echo
echo "Combined CSV lines, including header:"

wc -l "$COMBINED"

echo
echo "BIOSCAN metadata selection completed"
echo "No images were downloaded"
