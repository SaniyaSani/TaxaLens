#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-inat-replace
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
PLAN_DIR="$DATA_ROOT/pipeline/planning/training_plan_v1"
CANDIDATE="$DATA_ROOT/pipeline/image_download/inaturalist_replacement_candidate.json"
APPLIED="$DATA_ROOT/pipeline/image_download/inaturalist_replacement_applied.json"
BACKUP="$DATA_ROOT/pipeline/planning/pre_inat_replacement_${SLURM_JOB_ID}"

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

apptainer exec "$CONTAINER" "$VENV/bin/python" - \
    "$MASTER" \
    "$PLAN_DIR" \
    "$CANDIDATE" \
    "$APPLIED" \
    "$BACKUP" <<'PY'
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


master_path = Path(sys.argv[1])
plan_dir = Path(sys.argv[2])
candidate_path = Path(sys.argv[3])
applied_path = Path(sys.argv[4])
backup_dir = Path(sys.argv[5])

plan_path = plan_dir / "training_plan.parquet"
report_path = plan_dir / "training_plan_report.json"
groups_dir = plan_dir / "groups"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


candidate_audit = json.loads(
    candidate_path.read_text(encoding="utf-8")
)

old_id = str(candidate_audit["original"]["record_id"])
new_id = str(candidate_audit["replacement"]["record_id"])

print("Original record:", old_id)
print("Replacement:    ", new_id)

plan = pd.read_parquet(plan_path).fillna("")

old_rows = plan.index[
    plan["record_id"].astype(str).eq(old_id)
].tolist()

if len(old_rows) != 1:
    raise SystemExit(
        f"STOP: expected one original row in final plan; found {len(old_rows)}"
    )

if plan["record_id"].astype(str).eq(new_id).any():
    raise SystemExit("STOP: replacement is already present in final plan")

candidate_frame = pd.read_parquet(
    master_path,
    filters=[("record_id", "==", new_id)],
).fillna("")

if len(candidate_frame) != 1:
    raise SystemExit(
        f"STOP: expected one replacement row in master; found {len(candidate_frame)}"
    )

candidate = candidate_frame.iloc[0]
original_index = old_rows[0]
original = plan.loc[original_index].copy()

for field in ("source", "family", "genus", "species", "split"):
    expected = str(candidate_audit["replacement"].get(field, ""))
    actual = str(candidate.get(field, ""))

    if actual != expected:
        raise SystemExit(
            f"STOP: candidate {field} changed: "
            f"expected {expected!r}, found {actual!r}"
        )

if not str(candidate.get("image_url", "")):
    raise SystemExit("STOP: replacement has no image URL")

group_column = str(candidate_audit["group_column"])
new_group = str(candidate.get(group_column, ""))

if not new_group:
    raise SystemExit("STOP: replacement has no split group")

if new_group in set(plan[group_column].astype(str)):
    raise SystemExit(
        "STOP: replacement split group is already present in plan"
    )

group_paths = sorted(groups_dir.glob("*.parquet"))
matching_groups = []

for group_path in group_paths:
    ids = pd.read_parquet(
        group_path,
        columns=["record_id"],
    )["record_id"].astype(str)

    if ids.eq(old_id).any():
        matching_groups.append(group_path)

if len(matching_groups) != 1:
    raise SystemExit(
        "STOP: expected the original row in exactly one group plan; "
        f"found {len(matching_groups)}"
    )

group_path = matching_groups[0]
group = pd.read_parquet(group_path).fillna("")

if group["record_id"].astype(str).eq(new_id).any():
    raise SystemExit(
        "STOP: replacement is already present in group plan"
    )

group_indices = group.index[
    group["record_id"].astype(str).eq(old_id)
].tolist()

if len(group_indices) != 1:
    raise SystemExit(
        f"STOP: expected one group row; found {len(group_indices)}"
    )

group_index = group_indices[0]

print("Group plan:     ", group_path.name)

old_plan_sha = digest(plan_path)
old_group_sha = digest(group_path)

report = json.loads(
    report_path.read_text(encoding="utf-8")
)

if report.get("sha256") != old_plan_sha:
    raise SystemExit(
        "STOP: current training-plan digest does not match its report"
    )

backup_dir.mkdir(parents=True, exist_ok=False)

for source in (
    plan_path,
    report_path,
    group_path,
    candidate_path,
):
    shutil.copy2(source, backup_dir / source.name)

group_report_path = group_path.with_name(
    group_path.stem + "_report.json"
)

if group_report_path.is_file():
    shutil.copy2(
        group_report_path,
        backup_dir / group_report_path.name,
    )

print("Backup:         ", backup_dir)

plan_after = plan.copy()
group_after = group.copy()

# Keep planning-only slot metadata, but replace every canonical field
# available from the master-manifest candidate.
for column in plan_after.columns:
    if column in candidate.index:
        plan_after.at[original_index, column] = candidate[column]

for column in group_after.columns:
    if column in candidate.index:
        group_after.at[group_index, column] = candidate[column]

if len(plan_after) != 300_000:
    raise SystemExit(
        f"STOP: final plan has {len(plan_after)} rows instead of 300000"
    )

if plan_after["record_id"].astype(str).duplicated().any():
    raise SystemExit("STOP: duplicate record_id after replacement")

if plan_after["record_id"].astype(str).eq(old_id).any():
    raise SystemExit("STOP: deleted-image record remains in plan")

if plan_after["record_id"].astype(str).eq(new_id).sum() != 1:
    raise SystemExit("STOP: replacement is not present exactly once")

for column in (
    "source",
    "split",
    "planning_group",
    "family",
    "genus",
    "species",
    "identification_level",
):
    if column not in plan_after.columns:
        continue

    before = Counter(plan[column].astype(str))
    after = Counter(plan_after[column].astype(str))

    if before != after:
        raise SystemExit(
            f"STOP: aggregate counts changed for {column}"
        )

leaking = (
    plan_after[
        [group_column, "split"]
    ]
    .astype(str)
    .groupby(group_column)["split"]
    .nunique()
    .gt(1)
)

if leaking.any():
    raise SystemExit("STOP: split-group leakage after replacement")

all_group_ids = []

for current_path in group_paths:
    if current_path == group_path:
        ids = group_after["record_id"].astype(str).tolist()
    else:
        ids = pd.read_parquet(
            current_path,
            columns=["record_id"],
        )["record_id"].astype(str).tolist()

    all_group_ids.extend(ids)

plan_ids = plan_after["record_id"].astype(str).tolist()

if len(all_group_ids) != len(plan_ids):
    raise SystemExit(
        "STOP: merged group-plan row count differs from final plan"
    )

if set(all_group_ids) != set(plan_ids):
    raise SystemExit(
        "STOP: group plans and final plan contain different records"
    )

plan_pending = plan_path.with_name(
    "training_plan.pending-inat-replacement.parquet"
)
group_pending = group_path.with_name(
    group_path.stem + ".pending-inat-replacement.parquet"
)
report_pending = report_path.with_name(
    "training_plan_report.pending-inat-replacement.json"
)

plan_after.to_parquet(plan_pending, index=False)
group_after.to_parquet(group_pending, index=False)

new_plan_sha = digest(plan_pending)
new_group_sha = digest(group_pending)

report["sha256"] = new_plan_sha

report_pending.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

# All validation completed before any original artifact is replaced.
os.replace(group_pending, group_path)
os.replace(plan_pending, plan_path)
os.replace(report_pending, report_path)

applied = {
    "schema": 1,
    "applied_utc": datetime.now(timezone.utc).isoformat(),
    "candidate_audit": str(candidate_path),
    "backup": str(backup_dir),
    "group_plan": str(group_path),
    "rows": int(len(plan_after)),
    "old_training_plan_sha256": old_plan_sha,
    "new_training_plan_sha256": new_plan_sha,
    "old_group_plan_sha256": old_group_sha,
    "new_group_plan_sha256": new_group_sha,
    "original": candidate_audit["original"],
    "replacement": candidate_audit["replacement"],
    "validation": {
        "unique_record_ids": True,
        "source_counts_preserved": True,
        "split_counts_preserved": True,
        "taxonomy_counts_preserved": True,
        "planning_group_counts_preserved": True,
        "group_membership_consistent": True,
        "leaking_groups": 0,
    },
}

applied_pending = applied_path.with_suffix(".pending.json")
applied_pending.write_text(
    json.dumps(applied, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
os.replace(applied_pending, applied_path)

print()
print("Replacement applied successfully")
print("Rows:            ", f"{len(plan_after):,}")
print("Old plan SHA-256:", old_plan_sha)
print("New plan SHA-256:", new_plan_sha)
print("Applied report:  ", applied_path)
print("Original artifacts preserved in:", backup_dir)
PY

echo
echo "=== UPDATED PLAN FILES ==="

ls -lh \
    "$PLAN_DIR/training_plan.parquet" \
    "$PLAN_DIR/training_plan_report.json" \
    "$APPLIED"

echo
echo "iNaturalist replacement completed"
echo "No images were downloaded"
