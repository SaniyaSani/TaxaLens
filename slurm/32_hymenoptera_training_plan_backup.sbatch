#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-plan-backup
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"

PLAN_DIR="$DATA_ROOT/pipeline/planning/training_plan_v1"
REPORT="$PLAN_DIR/training_plan_report.json"
PLAN="$PLAN_DIR/training_plan.parquet"
JOB_ENV="$DATA_ROOT/pipeline/training_plan_job.env"

BACKUP_PARENT="/shares/hawlitschek.ieu.uzh/$USER/TaxaLens/v11_hymenoptera_teacher"
BACKUP="$BACKUP_PARENT/training_plan_v1"
STAGING="$BACKUP_PARENT/.training_plan_v1.pending-${SLURM_JOB_ID}"

cd "$REPO_DIR"

echo "Source plan: $PLAN_DIR"
echo "Final backup: $BACKUP"

for required in "$PLAN" "$REPORT"
do
    if [[ ! -s "$required" ]]
    then
        echo "STOP: required plan file is missing or empty: $required"
        exit 1
    fi
done

if [[ -e "$BACKUP" ]]
then
    echo "STOP: permanent training_plan_v1 already exists"
    echo "Existing checkpoint was not overwritten"
    exit 1
fi

if [[ -e "$STAGING" ]]
then
    echo "STOP: staging path already exists: $STAGING"
    exit 1
fi

mkdir -p \
    "$STAGING/plan" \
    "$STAGING/audit" \
    "$STAGING/config"

echo
echo "=== VERIFY PRIMARY PLAN DIGEST ==="

expected_sha=$(
    apptainer exec "$CONTAINER" "$VENV/bin/python" - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(
    Path(sys.argv[1]).read_text(encoding="utf-8")
)
print(report["sha256"])
PY
)

actual_sha=$(
    sha256sum "$PLAN" |
    awk '{print $1}'
)

if [[ "$actual_sha" != "$expected_sha" ]]
then
    echo "STOP: training plan digest mismatch"
    echo "Expected: $expected_sha"
    echo "Actual:   $actual_sha"
    exit 1
fi

echo "Training plan digest verified: $actual_sha"

echo
echo "=== COPY PLAN ==="

rsync -a \
    "$PLAN_DIR/" \
    "$STAGING/plan/"

echo
echo "=== COPY AUDIT OUTPUTS ==="

rsync -a \
    "$DATA_ROOT/pipeline/planning/master_planning_audit.json" \
    "$DATA_ROOT/pipeline/planning/family_by_source.tsv" \
    "$DATA_ROOT/pipeline/planning/family_source_split.tsv" \
    "$STAGING/audit/"

echo
echo "=== COPY VERSIONED CONFIGURATION ==="

rsync -a \
    "$REPO_DIR/configs/hymenoptera_teacher_300k_v1.json" \
    "$REPO_DIR/configs/hymenoptera_teacher_plan_v1" \
    "$STAGING/config/"

if [[ -s "$JOB_ENV" ]]
then
    cp "$JOB_ENV" "$STAGING/"
fi

git_commit=$(git rev-parse HEAD)
git_branch=$(git branch --show-current)
created_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
planning_job="unknown"

if [[ -s "$JOB_ENV" ]]
then
    source "$JOB_ENV"
    planning_job="${training_plan_job:-unknown}"
fi

cat > "$STAGING/PROVENANCE.txt" <<EOF
snapshot_id=hymenoptera_training_plan_v1
created_utc=$created_utc
git_commit=$git_commit
git_branch=$git_branch
planning_job=$planning_job
training_plan_rows=300000
training_plan_sha256=$actual_sha
bioscan_rows=154115
gbif_rows=85000
inaturalist_rows=60885
parasitoid_rows=235097
parasitoid_fraction=0.7836566666666667
contents=plan,group_plans,reports,audit,versioned_configs
EOF

echo
echo "=== CREATE AND VERIFY CHECKSUMS ==="

(
    cd "$STAGING"

    find . \
        -type f \
        ! -name 'SHA256SUMS.txt' \
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
echo "=== PERMANENT TRAINING PLAN ==="

du -sh "$BACKUP"
wc -l "$BACKUP/SHA256SUMS.txt"
cat "$BACKUP/PROVENANCE.txt"

echo
echo "Hymenoptera training-plan backup completed"
echo "Scratch files were not deleted"
