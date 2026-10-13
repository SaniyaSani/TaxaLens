#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-v10-data-backup
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
SCRATCH_ROOT="/scratch/$USER/taxalens"
RUN_ROOT="$SCRATCH_ROOT/v10_raw200k_swiss28"

BACKUP_PARENT="/shares/hawlitschek.ieu.uzh/$USER/TaxaLens/v10_raw200k_swiss28"
BACKUP="$BACKUP_PARENT/training_data_v1"
STAGING="$BACKUP_PARENT/.training_data_v1.pending-${SLURM_JOB_ID}"

if [[ -e "$BACKUP" ]]
then
    echo "STOP: final backup already exists: $BACKUP"
    exit 1
fi

required=(
    "$SCRATCH_ROOT/image_cache"
    "$SCRATCH_ROOT/source_cache/bioscan/diptera_60k_images"
    "$RUN_ROOT/images_quality_crops"
    "$RUN_ROOT/pipeline"
)

for source in "${required[@]}"
do
    if [[ ! -d "$source" ]]
    then
        echo "STOP: required directory is missing: $source"
        exit 1
    fi
done

mkdir -p \
    "$STAGING/image_cache" \
    "$STAGING/bioscan_images" \
    "$STAGING/images_quality_crops" \
    "$STAGING/pipeline"

echo "Compute node: $(hostname)"
echo "Staging: $STAGING"
echo "Final backup: $BACKUP"

echo
echo "=== COPY MULTISOURCE IMAGE CACHE ==="

rsync -a --info=stats2 \
    "$SCRATCH_ROOT/image_cache/" \
    "$STAGING/image_cache/"

echo
echo "=== COPY BIOSCAN IMAGES ==="

rsync -a --info=stats2 \
    "$SCRATCH_ROOT/source_cache/bioscan/diptera_60k_images/" \
    "$STAGING/bioscan_images/"

echo
echo "=== COPY QUALITY CROPS ==="

rsync -a --info=stats2 \
    "$RUN_ROOT/images_quality_crops/" \
    "$STAGING/images_quality_crops/"

echo
echo "=== COPY COMPLETE V10 PIPELINE ==="

rsync -a --info=stats2 \
    "$RUN_ROOT/pipeline/" \
    "$STAGING/pipeline/"

echo
echo "=== WRITE PROVENANCE ==="

{
    echo "snapshot_id=diptera_training_data_v1"
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$REPO_DIR" rev-parse HEAD)"
    echo "git_branch=$(git -C "$REPO_DIR" branch --show-current)"
    echo "source_run=$RUN_ROOT"
    echo "contents=image_cache,bioscan_images,images_quality_crops,pipeline"
} > "$STAGING/PROVENANCE.txt"

echo
echo "=== CREATE AND VERIFY CHECKSUMS ==="

(
    cd "$STAGING"

    find . \
        -type f \
        ! -name SHA256SUMS.txt \
        -print0 |
        sort -z |
        xargs -0 sha256sum \
        > SHA256SUMS.txt

    sha256sum --quiet -c SHA256SUMS.txt
)

echo "All checksums verified"

echo
echo "=== FINALIZE ATOMICALLY ==="

mv "$STAGING" "$BACKUP"

echo
echo "=== FINAL BACKUP ==="

du -sh "$BACKUP"
wc -l "$BACKUP/SHA256SUMS.txt"
cat "$BACKUP/PROVENANCE.txt"

echo
echo "Diptera training-data checkpoint completed"
echo "Scratch sources were not deleted"
