#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-plan-300k
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"

MASTER="$DATA_ROOT/pipeline/corpus/master_manifest.parquet"
CONFIG_DIR="$REPO_DIR/configs/hymenoptera_teacher_plan_v1"
MASTER_CONFIG="$REPO_DIR/configs/hymenoptera_teacher_300k_v1.json"

PLANNING_ROOT="$DATA_ROOT/pipeline/planning"
FINAL_DIR="$PLANNING_ROOT/training_plan_v1"
STAGING="$PLANNING_ROOT/.training_plan_v1.pending-${SLURM_JOB_ID}"
GROUP_DIR="$STAGING/groups"

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Master manifest: $MASTER"
echo "Final plan: $FINAL_DIR"

if [[ ! -s "$MASTER" ]]
then
    echo "STOP: master manifest is missing or empty: $MASTER"
    exit 1
fi

if [[ -e "$FINAL_DIR" ]]
then
    echo "STOP: a completed training_plan_v1 already exists"
    echo "Choose a new version instead of overwriting it"
    exit 1
fi

mkdir -p "$GROUP_DIR"

groups=(
    minute_parasitoid_enriched
    braconid_ichneumonid
    other_parasitoid_enriched
    aculeata_context
    mixed_and_symphyta_context
)

for group in "${groups[@]}"
do
    echo
    echo "=== PLAN GROUP: $group ==="

    apptainer exec "$CONTAINER" "$VENV/bin/python" \
        scripts/plan_foundation_corpus.py \
        --input "$MASTER" \
        --config "$CONFIG_DIR/${group}.json" \
        --out "$GROUP_DIR/${group}.parquet" \
        --report "$GROUP_DIR/${group}_report.json" \
        --chunksize 100000
done

echo
echo "=== MERGE AND VALIDATE GROUP PLANS ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" - \
    "$MASTER" \
    "$MASTER_CONFIG" \
    "$CONFIG_DIR" \
    "$STAGING" <<'PYMERGE'
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

master_path = Path(sys.argv[1])
master_config_path = Path(sys.argv[2])
config_dir = Path(sys.argv[3])
staging = Path(sys.argv[4])
group_dir = staging / "groups"

master_config = json.loads(
    master_config_path.read_text(encoding="utf-8")
)

frames = []
expected_total = int(master_config["total_images"])
expected_sources = Counter()
expected_groups = {}

for group_spec in master_config["groups"]:
    name = group_spec["name"]
    expected = {
        str(source): int(count)
        for source, count in group_spec["source_targets"].items()
    }
    expected_sources.update(expected)
    expected_groups[name] = sum(expected.values())

    config = json.loads(
        (config_dir / f"{name}.json").read_text(encoding="utf-8")
    )
    scope_path = Path(config["target_families_file"])

    if not scope_path.is_absolute():
        scope_path = Path.cwd() / scope_path

    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    allowed_families = {
        str(value).strip().casefold()
        for value in scope["families"]
        if str(value).strip()
    }

    path = group_dir / f"{name}.parquet"
    frame = pd.read_parquet(path).fillna("")

    actual = {
        str(source): int(count)
        for source, count in frame["source"].value_counts().items()
    }

    if actual != expected:
        raise SystemExit(
            f"STOP: source quotas differ for {name}: "
            f"expected={expected}, actual={actual}"
        )

    unexpected_families = sorted({
        str(value)
        for value in frame["family"]
        if str(value).casefold() not in allowed_families
    })

    if unexpected_families:
        raise SystemExit(
            f"STOP: out-of-scope families in {name}: "
            + ", ".join(unexpected_families[:20])
        )

    frame["planning_group"] = name
    frames.append(frame)

combined = pd.concat(frames, ignore_index=True)

if len(combined) != expected_total:
    raise SystemExit(
        f"STOP: expected {expected_total:,} rows, "
        f"found {len(combined):,}"
    )

if combined["record_id"].astype(str).eq("").any():
    raise SystemExit("STOP: empty record_id values in training plan")

duplicate_ids = combined["record_id"].astype(str).duplicated(
    keep=False
)

if duplicate_ids.any():
    examples = sorted(
        combined.loc[duplicate_ids, "record_id"]
        .astype(str)
        .unique()
    )[:20]
    raise SystemExit(
        "STOP: record_id overlap between group plans: "
        + ", ".join(examples)
    )

actual_sources = Counter(
    combined["source"].astype(str)
)

if actual_sources != expected_sources:
    raise SystemExit(
        f"STOP: combined source totals differ: "
        f"expected={dict(expected_sources)}, "
        f"actual={dict(actual_sources)}"
    )

actual_groups = {
    str(name): int(count)
    for name, count in combined[
        "planning_group"
    ].value_counts().items()
}

if actual_groups != expected_groups:
    raise SystemExit(
        f"STOP: combined group totals differ: "
        f"expected={expected_groups}, actual={actual_groups}"
    )

unknown_splits = (
    set(combined["split"].astype(str))
    - {"train", "val", "test"}
)

if unknown_splits:
    raise SystemExit(
        "STOP: unknown dataset splits: "
        + ", ".join(sorted(unknown_splits))
    )

group_column = next(
    (
        name
        for name in (
            "split_group",
            "duplicate_group_id",
            "specimen_group_id",
            "record_id",
        )
        if name in combined.columns
    ),
    None,
)

if not group_column:
    raise SystemExit(
        "STOP: no group identifier for leakage validation"
    )

groups = combined[group_column].astype(str)

if groups.eq("").any():
    groups = groups.where(
        groups.ne(""),
        "record:" + combined["record_id"].astype(str),
    )

leaking = (
    pd.DataFrame({
        "group": groups,
        "split": combined["split"].astype(str),
    })
    .groupby("group")["split"]
    .nunique()
    .gt(1)
)

if leaking.any():
    raise SystemExit(
        f"STOP: {int(leaking.sum())} split groups leak "
        "across dataset splits"
    )

master_ids = set(
    pd.read_parquet(
        master_path,
        columns=["record_id", "source"],
    )
    .query("source == 'BIOSCAN-5M'")["record_id"]
    .astype(str)
)

selected_bioscan_ids = set(
    combined.loc[
        combined["source"].eq("BIOSCAN-5M"),
        "record_id",
    ].astype(str)
)

if selected_bioscan_ids != master_ids:
    raise SystemExit(
        "STOP: the plan does not retain exactly all BIOSCAN rows; "
        f"missing={len(master_ids - selected_bioscan_ids)}, "
        f"unexpected={len(selected_bioscan_ids - master_ids)}"
    )

combined = combined.sort_values(
    ["planning_group", "source", "record_id"],
    kind="stable",
).reset_index(drop=True)

output = staging / "training_plan.parquet"
pending = staging / "training_plan.pending.parquet"
combined.to_parquet(pending, index=False)
os.replace(pending, output)

digest = hashlib.sha256()

with output.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)

parasitoid_groups = set(
    master_config["parasitoid_groups"]
)
parasitoid_rows = int(
    combined["planning_group"]
    .isin(parasitoid_groups)
    .sum()
)

report = {
    "schema": 1,
    "profile": master_config["profile"],
    "rows": int(len(combined)),
    "sha256": digest.hexdigest(),
    "source_counts": {
        str(key): int(value)
        for key, value in combined[
            "source"
        ].value_counts().items()
    },
    "planning_group_counts": {
        str(key): int(value)
        for key, value in combined[
            "planning_group"
        ].value_counts().items()
    },
    "split_counts": {
        str(key): int(value)
        for key, value in combined[
            "split"
        ].value_counts().items()
    },
    "identification_level_counts": {
        str(key): int(value)
        for key, value in combined[
            "identification_level"
        ].value_counts().items()
    },
    "parasitoid_rows": parasitoid_rows,
    "parasitoid_fraction": (
        parasitoid_rows / len(combined)
    ),
    "bioscan_rows": len(selected_bioscan_ids),
    "all_bioscan_retained": True,
    "group_column": group_column,
    "leaking_groups": 0,
    "outside_scope_policy": master_config[
        "outside_scope_policy"
    ],
}

report_path = staging / "training_plan_report.json"
report_pending = staging / "training_plan_report.pending.json"
report_pending.write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(report_pending, report_path)

print(json.dumps(report, indent=2))
PYMERGE

echo
echo "=== FINALIZE PLAN ATOMICALLY ==="

mv "$STAGING" "$FINAL_DIR"

echo
echo "=== FINAL TRAINING PLAN ==="

ls -lh \
    "$FINAL_DIR/training_plan.parquet" \
    "$FINAL_DIR/training_plan_report.json" \
    "$FINAL_DIR/groups"/*.parquet \
    "$FINAL_DIR/groups"/*_report.json

echo
echo "Hymenoptera 300k training plan completed"
echo "No images were downloaded"
