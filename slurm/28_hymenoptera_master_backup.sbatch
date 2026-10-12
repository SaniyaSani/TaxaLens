#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-final-backup
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-/home/$USER/projects/TaxaLens}"
SOURCE_ROOT="/scratch/$USER/taxalens/v11_hymenoptera_teacher"
BACKUP_PARENT="/shares/hawlitschek.ieu.uzh/$USER/TaxaLens/v11_hymenoptera_teacher"
BACKUP_ROOT="$BACKUP_PARENT/final_corpus_metadata_v1"
STAGING="$BACKUP_PARENT/.final_corpus_metadata_v1.pending-${SLURM_JOB_ID}"

files=(
    pipeline/corpus/bioscan_raw.parquet
    pipeline/corpus/master_deduplicated.parquet
    pipeline/corpus/master_manifest.parquet
    pipeline/master_corpus_report.json
    pipeline/master_corpus_report.md
    pipeline/bioscan_fix_job.env
    pipeline/assemble_job.env
    pipeline/master_report_job.env
)

echo "Compute node: $(hostname)"
echo "Source: $SOURCE_ROOT"
echo "Backup: $BACKUP_ROOT"

if [[ -e "$BACKUP_ROOT" ]]
then
    echo "STOP: backup already exists: $BACKUP_ROOT"
    exit 1
fi

mkdir -p "$BACKUP_PARENT" "$STAGING"

for file in "${files[@]}"
do
    if [[ ! -f "$SOURCE_ROOT/$file" ]]
    then
        echo "STOP: required file is missing: $SOURCE_ROOT/$file"
        exit 1
    fi
done

echo
echo "Copying final corpus metadata..."

(
    cd "$SOURCE_ROOT"
    rsync -aR --info=progress2 "${files[@]}" "$STAGING/"
)

source "$SOURCE_ROOT/pipeline/bioscan_fix_job.env"
source "$SOURCE_ROOT/pipeline/assemble_job.env"
source "$SOURCE_ROOT/pipeline/master_report_job.env"

{
    echo "snapshot_id=hymenoptera_master_metadata_v1"
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$REPO_DIR" rev-parse HEAD)"
    echo "git_branch=$(git -C "$REPO_DIR" branch --show-current)"
    echo "bioscan_normalization_job=$bioscan_fix_job"
    echo "assembly_job=$assemble_job"
    echo "report_job=$master_report_job"
    echo "master_rows=2191556"
    echo "bioscan_rows=154115"
} > "$STAGING/PROVENANCE.txt"

echo
echo "Creating checksums..."

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

mv "$STAGING" "$BACKUP_ROOT"

echo
echo "Final corpus metadata backup completed and verified"

du -sh "$BACKUP_ROOT"
wc -l "$BACKUP_ROOT/SHA256SUMS.txt"
cat "$BACKUP_ROOT/PROVENANCE.txt"
