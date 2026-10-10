#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-backup
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

SOURCE_ROOT="/scratch/$USER/taxalens/v11_hymenoptera_teacher"
BACKUP_ROOT="/shares/hawlitschek.ieu.uzh/$USER/TaxaLens/v11_hymenoptera_teacher/source_snapshot"

mkdir -p "$BACKUP_ROOT"

cd "$SOURCE_ROOT"

required_files=(
    pipeline/corpus/inat_raw.parquet
    pipeline/corpus/gbif_raw.parquet
    pipeline/corpus/bioscan_selected.csv
    source/gbif_hymenoptera_0018027-260928105237408.zip
)

echo "Checking required files..."

for file in "${required_files[@]}"
do
    if [[ ! -f "$file" ]]
    then
        echo "STOP: required file is missing: $SOURCE_ROOT/$file"
        exit 1
    fi
done

echo
echo "Copying normalized source manifests..."

rsync -a --info=progress2 --relative \
    "${required_files[@]}" \
    "$BACKUP_ROOT/"

echo
echo "Copying census, BIOSCAN selections, requests and job metadata..."

rsync -a --info=progress2 --relative \
    pipeline/bioscan_selection/ \
    pipeline/census/ \
    pipeline/requests/ \
    "$BACKUP_ROOT/"

if compgen -G "pipeline/*.env" >/dev/null
then
    rsync -a --relative \
        pipeline/*.env \
        "$BACKUP_ROOT/"
fi

echo
echo "Creating checksum manifest..."

checksum_pending="$BACKUP_ROOT/SHA256SUMS.txt.pending"

(
    cd "$BACKUP_ROOT"

    find pipeline source \
        -type f \
        ! -name 'SHA256SUMS.txt' \
        ! -name 'SHA256SUMS.txt.pending' \
        -print0 |
        sort -z |
        xargs -0 sha256sum \
        > "$checksum_pending"

    mv "$checksum_pending" SHA256SUMS.txt

    echo
    echo "Verifying copied files..."

    sha256sum --quiet -c SHA256SUMS.txt
)

echo
echo "Hymenoptera source snapshot completed and verified"

du -sh "$BACKUP_ROOT"
wc -l "$BACKUP_ROOT/SHA256SUMS.txt"
