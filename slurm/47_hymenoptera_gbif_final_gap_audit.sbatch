#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-final-gap-audit
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
IMAGE_ROOT="${IMAGE_ROOT:-/scratch/$USER/taxalens/image_cache}"

MASTER="$DATA_ROOT/pipeline/corpus/master_manifest.parquet"
PLAN="$DATA_ROOT/pipeline/planning/training_plan_v1/training_plan.parquet"

AUDIT_DIR="$DATA_ROOT/pipeline/image_download/gbif_recovery_v1/laji_replacement_audit"
GAPS="$AUDIT_DIR/cross_source_v4/cross_source_unmatched_v4.parquet"
V3="$AUDIT_DIR/laji_replacement_candidates_v3.parquet"
V4="$AUDIT_DIR/cross_source_v4/cross_source_candidates_v4.parquet"

OUT_DIR="$AUDIT_DIR/same_group_v5"
CANDIDATES="$OUT_DIR/same_group_candidates_v5.parquet"
MAPPING="$OUT_DIR/same_group_mapping_v5.parquet"
UNMATCHED="$OUT_DIR/same_group_unmatched_v5.parquet"
REPORT="$OUT_DIR/same_group_audit_v5.json"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Remaining targets: $GAPS"
echo "No training-plan or image files will be modified"
echo

for file in "$MASTER" "$PLAN" "$GAPS" "$V3" "$V4"
do
    if [[ ! -s "$file" ]]
    then
        echo "STOP: required input is missing or empty: $file"
        exit 1
    fi
done

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - \
    "$MASTER" \
    "$PLAN" \
    "$GAPS" \
    "$V3" \
    "$V4" \
    "$IMAGE_ROOT" \
    "$CANDIDATES" \
    "$MAPPING" \
    "$UNMATCHED" \
    "$REPORT" <<'PY'
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


(
    master_path,
    plan_path,
    gaps_path,
    v3_path,
    v4_path,
    image_root,
    candidates_path,
    mapping_path,
    unmatched_path,
    report_path,
) = map(Path, sys.argv[1:])


def normalize(frame):
    for column in (
        "record_id",
        "source",
        "image_url",
        "image_license",
        "family",
        "genus",
        "species",
        "split",
        "split_group",
        "local_path",
    ):
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str)
    return frame


def require(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise SystemExit(
            f"STOP: {label} is missing columns: "
            + ", ".join(sorted(missing))
        )


def effective_groups(frame):
    values = frame["split_group"].fillna("").astype(str)
    return values.where(
        values.ne(""),
        "record:" + frame["record_id"].astype(str),
    )


def safe_source(value):
    return (
        re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-")
        or "unknown"
    )


def stable_hash(*values):
    return hashlib.sha256(
        "|".join(map(str, values)).encode("utf-8")
    ).hexdigest()


def atomic_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    frame.to_parquet(pending, index=False)
    pending.replace(path)


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


master = normalize(pd.read_parquet(master_path))
plan = normalize(pd.read_parquet(plan_path))
targets = normalize(pd.read_parquet(gaps_path))
v3 = normalize(pd.read_parquet(v3_path))
v4 = normalize(pd.read_parquet(v4_path))

required = {
    "record_id",
    "source",
    "image_url",
    "image_license",
    "family",
    "genus",
    "species",
    "split",
    "split_group",
}

for frame, label in (
    (master, "master"),
    (plan, "plan"),
    (targets, "targets"),
    (v3, "V3 candidates"),
    (v4, "V4 candidates"),
):
    require(frame, required, label)

if len(targets) != 44:
    raise SystemExit(
        f"STOP: expected 44 remaining targets; found {len(targets):,}"
    )

plan["_effective_group"] = effective_groups(plan)
v3["_effective_group"] = effective_groups(v3)
v4["_effective_group"] = effective_groups(v4)

group_splits = (
    plan[["_effective_group", "split"]]
    .drop_duplicates()
    .groupby("_effective_group")["split"]
    .agg(list)
)

leaking = group_splits[group_splits.map(len).gt(1)]

if not leaking.empty:
    raise SystemExit(
        f"STOP: current plan already has {len(leaking):,} leaking groups"
    )

plan_group_split = {
    group: splits[0]
    for group, splits in group_splits.items()
}

excluded_ids = (
    set(plan["record_id"])
    | set(v3["record_id"])
    | set(v4["record_id"])
)

excluded_urls = (
    set(plan.loc[plan["image_url"].ne(""), "image_url"])
    | set(v3.loc[v3["image_url"].ne(""), "image_url"])
    | set(v4.loc[v4["image_url"].ne(""), "image_url"])
)

reserved_groups = (
    set(v3["_effective_group"])
    | set(v4["_effective_group"])
)

target_families = set(targets["family"]) - {""}

eligible = pd.Series(True, index=master.index)

if "eligible_supervised" in master.columns:
    values = master["eligible_supervised"]

    if values.dtype == bool:
        eligible = values.fillna(False)
    else:
        eligible = (
            values.fillna("")
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes"})
        )

hosts = (
    master["image_url"]
    .map(lambda value: (urlparse(value).hostname or "").lower())
)

pool = master[
    master["source"].isin({"GBIF", "iNaturalist"})
    & master["family"].isin(target_families)
    & master["image_url"].ne("")
    & master["image_license"].isin({"CC0", "CC-BY", "CC-BY-SA"})
    & hosts.ne("image.laji.fi")
    & eligible
].copy()

pool["_effective_group"] = effective_groups(pool)

pool = pool[
    ~pool["record_id"].isin(excluded_ids)
    & ~pool["image_url"].isin(excluded_urls)
    & ~pool["_effective_group"].isin(reserved_groups)
].copy()

pool = pool.drop_duplicates("record_id").reset_index(drop=True)

pool["_group_plan_split"] = (
    pool["_effective_group"]
    .map(plan_group_split)
    .fillna("")
)

pool["_group_already_in_plan"] = pool["_group_plan_split"].ne("")


def cached_path(row):
    record_id = str(row["record_id"])
    return (
        image_root
        / safe_source(str(row["source"]))
        / record_id[:2]
        / f"{record_id}.jpg"
    )


pool["_cached"] = pool.apply(
    lambda row: (
        cached_path(row).is_file()
        and cached_path(row).stat().st_size > 0
    ),
    axis=1,
)

print(f"Remaining targets: {len(targets):,}")
print(f"Eligible same-family candidates: {len(pool):,}")
print(
    "Candidates from groups already in plan:",
    f"{int(pool['_group_already_in_plan'].sum()):,}",
)

targets["_order"] = targets["record_id"].map(
    lambda value: stable_hash("v5", "42", value)
)
targets = targets.sort_values("_order").reset_index(drop=True)

selected_rows = []
mapping_rows = []
unmatched_rows = []

used_ids = set()
used_urls = set()
used_groups = set()

for _, target in targets.iterrows():
    candidates = pool[
        pool["family"].eq(target["family"])
        & ~pool["record_id"].isin(used_ids)
        & ~pool["image_url"].isin(used_urls)
        & ~pool["_effective_group"].isin(used_groups)
    ].copy()

    if candidates.empty:
        unmatched_rows.append(
            target.drop(labels=["_order"]).to_dict()
        )
        continue

    candidates["_match_score"] = 1

    same_genus = (
        target["genus"] != ""
    ) & candidates["genus"].eq(target["genus"])

    same_species = (
        target["species"] != ""
    ) & candidates["species"].eq(target["species"])

    candidates.loc[same_genus, "_match_score"] = 2
    candidates.loc[same_species, "_match_score"] = 3

    candidates["_assigned_split"] = candidates.apply(
        lambda row: (
            row["_group_plan_split"]
            if row["_group_plan_split"]
            else target["split"]
        ),
        axis=1,
    )

    candidates["_same_target_split"] = (
        candidates["_assigned_split"].eq(target["split"])
    )

    candidates["_new_group"] = (
        ~candidates["_group_already_in_plan"]
    )

    candidates["_stable"] = candidates["record_id"].map(
        lambda value: stable_hash(
            "v5",
            "42",
            target["record_id"],
            value,
        )
    )

    candidates = candidates.sort_values(
        [
            "_match_score",
            "_new_group",
            "_same_target_split",
            "_cached",
            "_stable",
        ],
        ascending=[False, False, False, False, True],
    )

    chosen = candidates.iloc[0].copy()

    score = int(chosen["_match_score"])
    match_level = {
        3: "species",
        2: "genus",
        1: "family",
    }[score]

    assigned_split = str(chosen["_assigned_split"])
    original_split = str(chosen["split"])
    group = str(chosen["_effective_group"])
    already_in_plan = bool(chosen["_group_already_in_plan"])
    was_cached = bool(chosen["_cached"])

    chosen["split"] = assigned_split

    if was_cached and "local_path" in chosen.index:
        chosen["local_path"] = str(cached_path(chosen))

    if "planning_group" in target.index:
        chosen["planning_group"] = target["planning_group"]

    clean = chosen.drop(
        labels=[
            "_effective_group",
            "_group_plan_split",
            "_group_already_in_plan",
            "_cached",
            "_match_score",
            "_assigned_split",
            "_same_target_split",
            "_new_group",
            "_stable",
        ],
        errors="ignore",
    )

    selected_rows.append(clean.to_dict())

    mapping_rows.append({
        "target_record_id": target["record_id"],
        "replacement_record_id": chosen["record_id"],
        "target_source": target["source"],
        "replacement_source": chosen["source"],
        "family": target["family"],
        "match_level": match_level,
        "target_genus": target["genus"],
        "replacement_genus": chosen["genus"],
        "target_species": target["species"],
        "replacement_species": chosen["species"],
        "target_split": target["split"],
        "assigned_split": assigned_split,
        "candidate_original_split": original_split,
        "replacement_split_group": group,
        "group_already_in_plan": already_in_plan,
        "replacement_already_cached": was_cached,
        "replacement_image_url": chosen["image_url"],
    })

    used_ids.add(str(chosen["record_id"]))
    used_urls.add(str(chosen["image_url"]))
    used_groups.add(group)

selected = (
    pd.DataFrame(selected_rows)
    if selected_rows
    else master.iloc[:0].copy()
)

mapping = pd.DataFrame(mapping_rows)

unmatched = (
    pd.DataFrame(unmatched_rows)
    if unmatched_rows
    else targets.drop(columns=["_order"]).iloc[:0].copy()
)

split_changes = Counter(
    f"{row['target_split']}->{row['assigned_split']}"
    for row in mapping_rows
    if row["target_split"] != row["assigned_split"]
)

report = {
    "schema": 5,
    "policy": (
        "Final read-only feasibility audit. Allow an unused image "
        "record from a same-family specimen/observation group already "
        "present in the plan. Such a replacement must remain in that "
        "group's existing split, preventing cross-split leakage. "
        "Candidate records and URLs remain unique."
    ),
    "target_rows": len(targets),
    "candidate_pool_rows": len(pool),
    "mapped_rows": len(mapping),
    "unmatched_rows": len(unmatched),
    "mapped_from_existing_plan_groups": int(
        mapping["group_already_in_plan"].sum()
    ) if not mapping.empty else 0,
    "already_cached_replacements": int(
        mapping["replacement_already_cached"].sum()
    ) if not mapping.empty else 0,
    "downloads_needed": int(
        (~mapping["replacement_already_cached"]).sum()
    ) if not mapping.empty else 0,
    "match_levels": dict(Counter(mapping["match_level"]))
    if not mapping.empty else {},
    "replacement_sources": dict(
        Counter(mapping["replacement_source"])
    ) if not mapping.empty else {},
    "split_reassignments": dict(split_changes),
    "remaining_families": dict(
        Counter(unmatched["family"])
    ) if not unmatched.empty else {},
    "training_plan_modified": False,
    "images_downloaded": False,
}

atomic_parquet(selected, candidates_path)
atomic_parquet(mapping, mapping_path)
atomic_parquet(unmatched, unmatched_path)
atomic_json(report, report_path)

print()
print(json.dumps(report, indent=2, ensure_ascii=False))
print()
print(f"Candidates     -> {candidates_path}")
print(f"Mapping        -> {mapping_path}")
print(f"Remaining gaps -> {unmatched_path}")
print(f"Report         -> {report_path}")
PY

echo
echo "=== V5 OUTPUTS ==="

ls -lh \
    "$CANDIDATES" \
    "$MAPPING" \
    "$UNMATCHED" \
    "$REPORT"

echo
echo "Final gap feasibility audit completed"
echo "No training-plan or image files were modified"
