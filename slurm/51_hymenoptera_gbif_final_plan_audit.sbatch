#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-final-plan-audit
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"

PLAN="$DATA_ROOT/pipeline/planning/training_plan_v1/training_plan.parquet"
RECOVERY="$DATA_ROOT/pipeline/image_download/gbif_recovery_v1"
PARALLEL="$DATA_ROOT/pipeline/image_download/gbif_parallel_recovery_v1/final"
ROUND2="$DATA_ROOT/pipeline/image_download/gbif_round2_download_v1/final"
AUDIT_DIR="$DATA_ROOT/pipeline/image_download/gbif_final_plan_audit_v1"

mkdir -p "$AUDIT_DIR"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Current plan: $PLAN"
echo "No training-plan or image files will be modified"
echo

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - "$PLAN" "$RECOVERY" "$PARALLEL" "$ROUND2" "$AUDIT_DIR" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

plan_path = Path(sys.argv[1])
recovery = Path(sys.argv[2])
parallel = Path(sys.argv[3])
round2 = Path(sys.argv[4])
audit_dir = Path(sys.argv[5])

paths = {
    "plan": plan_path,
    "initial_valid": recovery / "gbif_valid.parquet",
    "initial_missing": recovery / "gbif_missing.parquet",
    "original_retry_valid": parallel / "original_retry_valid.parquet",
    "round1_replacement_valid": parallel / "replacement_valid.parquet",
    "v3_mapping": (
        recovery
        / "laji_replacement_audit"
        / "laji_replacement_mapping_v3.parquet"
    ),
    "v4_mapping": (
        recovery
        / "laji_replacement_audit"
        / "cross_source_v4"
        / "cross_source_mapping_v4.parquet"
    ),
    "round2_mapping": round2 / "round2_selected_mapping.parquet",
    "round2_replacements": round2 / "round2_selected_replacements.parquet",
    "round2_unresolved": round2 / "round2_unresolved_targets.parquet",
}

for name, path in paths.items():
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"STOP: missing or empty {name}: {path}")

frames = {
    name: pd.read_parquet(path)
    for name, path in paths.items()
}

def pick(frame, names, label):
    for name in names:
        if name in frame.columns:
            return name
    raise SystemExit(
        f"STOP: cannot identify {label}; columns are "
        + ", ".join(map(str, frame.columns))
    )

def ids(frame, column="record_id"):
    if column not in frame:
        raise SystemExit(f"STOP: missing {column}")
    values = frame[column].fillna("").astype(str)
    if values.eq("").any():
        raise SystemExit(f"STOP: empty values in {column}")
    return set(values)

def mapping_ids(frame):
    target_column = pick(
        frame,
        [
            "target_record_id",
            "original_record_id",
            "failed_record_id",
        ],
        "mapping target column",
    )
    candidate_column = pick(
        frame,
        [
            "candidate_record_id",
            "replacement_record_id",
            "selected_record_id",
        ],
        "mapping candidate column",
    )
    return (
        target_column,
        candidate_column,
        set(frame[target_column].fillna("").astype(str)),
        set(frame[candidate_column].fillna("").astype(str)),
    )

plan = frames["plan"]
initial_valid = frames["initial_valid"]
initial_missing = frames["initial_missing"]
retry_valid = frames["original_retry_valid"]
round1_valid = frames["round1_replacement_valid"]
round2_replacements = frames["round2_replacements"]
round2_unresolved = frames["round2_unresolved"]

if len(plan) != 300_000:
    raise SystemExit(
        f"STOP: expected 300,000 plan rows, found {len(plan):,}"
    )

plan_ids = ids(plan)
initial_valid_ids = ids(initial_valid)
initial_missing_ids = ids(initial_missing)
retry_ids = ids(retry_valid)
round1_candidate_ids = ids(round1_valid)
round2_candidate_ids = ids(round2_replacements)

v3 = mapping_ids(frames["v3_mapping"])
v4 = mapping_ids(frames["v4_mapping"])
r2 = mapping_ids(frames["round2_mapping"])

round1_target_by_candidate = {}

for frame, columns in (
    (frames["v3_mapping"], v3),
    (frames["v4_mapping"], v4),
):
    target_column, candidate_column = columns[:2]

    for target, candidate in zip(
        frame[target_column].fillna("").astype(str),
        frame[candidate_column].fillna("").astype(str),
    ):
        previous = round1_target_by_candidate.get(candidate)

        if previous is not None and previous != target:
            raise SystemExit(
                f"STOP: candidate {candidate} maps to multiple targets"
            )

        round1_target_by_candidate[candidate] = target

missing_round1_mappings = (
    round1_candidate_ids - set(round1_target_by_candidate)
)

if missing_round1_mappings:
    raise SystemExit(
        "STOP: successful round-1 candidates without target mappings: "
        f"{len(missing_round1_mappings):,}"
    )

round1_recovered_targets = {
    round1_target_by_candidate[candidate]
    for candidate in round1_candidate_ids
}

r2_target_column, r2_candidate_column = r2[:2]

round2_target_by_candidate = dict(
    zip(
        frames["round2_mapping"][r2_candidate_column]
        .fillna("")
        .astype(str),
        frames["round2_mapping"][r2_target_column]
        .fillna("")
        .astype(str),
    )
)

missing_round2_mappings = (
    round2_candidate_ids - set(round2_target_by_candidate)
)

if missing_round2_mappings:
    raise SystemExit(
        "STOP: successful round-2 candidates without target mappings: "
        f"{len(missing_round2_mappings):,}"
    )

round2_recovered_targets = {
    round2_target_by_candidate[candidate]
    for candidate in round2_candidate_ids
}

unresolved_target_column = pick(
    round2_unresolved,
    [
        "target_record_id",
        "original_record_id",
        "failed_record_id",
        "record_id",
    ],
    "unresolved target column",
)

unresolved_ids = set(
    round2_unresolved[unresolved_target_column]
    .fillna("")
    .astype(str)
)

resolved_sets = {
    "original_retry": retry_ids,
    "round1_replacement": round1_recovered_targets,
    "round2_replacement": round2_recovered_targets,
}

names = list(resolved_sets)

for index, left_name in enumerate(names):
    for right_name in names[index + 1:]:
        overlap = (
            resolved_sets[left_name]
            & resolved_sets[right_name]
        )
        if overlap:
            raise SystemExit(
                f"STOP: {left_name} and {right_name} overlap: "
                f"{len(overlap):,}"
            )

resolved_missing_ids = set().union(*resolved_sets.values())
calculated_unresolved = initial_missing_ids - resolved_missing_ids

if not resolved_missing_ids <= initial_missing_ids:
    raise SystemExit(
        "STOP: recovered targets occur outside the initial missing set"
    )

if calculated_unresolved != unresolved_ids:
    raise SystemExit(
        "STOP: unresolved-target set does not match recovery arithmetic: "
        f"calculated={len(calculated_unresolved):,}, "
        f"reported={len(unresolved_ids):,}"
    )

if initial_valid_ids & initial_missing_ids:
    raise SystemExit("STOP: initial valid and missing sets overlap")

if (initial_valid_ids | initial_missing_ids) != set(
    plan.loc[
        plan["source"].fillna("").astype(str).eq("GBIF"),
        "record_id",
    ].astype(str)
):
    raise SystemExit(
        "STOP: initial GBIF audit does not partition the plan"
    )

replacement_ids = round1_candidate_ids | round2_candidate_ids

if replacement_ids & plan_ids:
    raise SystemExit(
        "STOP: replacement record IDs already occur in the current plan"
    )

if round1_candidate_ids & round2_candidate_ids:
    raise SystemExit(
        "STOP: round-1 and round-2 candidates overlap"
    )

# The current plan already contains all initially valid GBIF rows.
# Remove the full initial-missing set, then add only recovered targets.
projected_rows = (
    len(plan)
    - len(initial_missing_ids)
    + len(retry_ids)
    + len(round1_candidate_ids)
    + len(round2_candidate_ids)
)

report = {
    "schema": 1,
    "current_plan_rows": int(len(plan)),
    "initial_gbif_valid": int(len(initial_valid_ids)),
    "initial_gbif_missing": int(len(initial_missing_ids)),
    "original_retry_recovered": int(len(retry_ids)),
    "round1_replacements_recovered": int(
        len(round1_candidate_ids)
    ),
    "round2_replacements_recovered": int(
        len(round2_candidate_ids)
    ),
    "total_missing_targets_recovered": int(
        len(resolved_missing_ids)
    ),
    "final_unresolved_targets": int(len(unresolved_ids)),
    "projected_final_rows": int(projected_rows),
    "round1_replacement_sources": {
        str(key): int(value)
        for key, value in round1_valid["source"]
        .fillna("")
        .astype(str)
        .value_counts()
        .items()
    },
    "round2_replacement_sources": {
        str(key): int(value)
        for key, value in round2_replacements["source"]
        .fillna("")
        .astype(str)
        .value_counts()
        .items()
    },
    "plan_columns": list(map(str, plan.columns)),
    "round1_missing_plan_columns": sorted(
        set(plan.columns) - set(round1_valid.columns)
    ),
    "round2_missing_plan_columns": sorted(
        set(plan.columns) - set(round2_replacements.columns)
    ),
    "round1_extra_columns": sorted(
        set(round1_valid.columns) - set(plan.columns)
    ),
    "round2_extra_columns": sorted(
        set(round2_replacements.columns) - set(plan.columns)
    ),
    "target_sets_disjoint": True,
    "replacement_ids_absent_from_current_plan": True,
    "unresolved_set_verified": True,
    "training_plan_modified": False,
}

if projected_rows != 298_932:
    raise SystemExit(
        f"STOP: projected rows are {projected_rows:,}, "
        "expected 298,932"
    )

audit_dir.mkdir(parents=True, exist_ok=True)
report_path = audit_dir / "final_plan_input_audit.json"
pending = report_path.with_name(report_path.name + ".pending")
pending.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
pending.replace(report_path)

print(json.dumps(report, indent=2, ensure_ascii=False))
print()
print(f"Report -> {report_path}")
print("Final-plan inputs are internally consistent")
print("No training-plan files were modified")
PY

echo
echo "=== AUDIT OUTPUT ==="

ls -lh "$AUDIT_DIR/final_plan_input_audit.json"

echo
echo "GBIF final-plan input audit completed"
