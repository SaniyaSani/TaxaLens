#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-gbif-source-audit
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

GAPS="$AUDIT_DIR/laji_unmatched_targets_v3.parquet"
V3_CANDIDATES="$AUDIT_DIR/laji_replacement_candidates_v3.parquet"

OUT_DIR="$AUDIT_DIR/cross_source_v4"
CANDIDATES="$OUT_DIR/cross_source_candidates_v4.parquet"
MAPPING="$OUT_DIR/cross_source_mapping_v4.parquet"
UNMATCHED="$OUT_DIR/cross_source_unmatched_v4.parquet"
REPORT="$OUT_DIR/cross_source_audit_v4.json"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Remaining GBIF targets: $GAPS"
echo "Fallback source: iNaturalist"
echo "No training-plan or image files will be modified"
echo

for file in "$MASTER" "$PLAN" "$GAPS" "$V3_CANDIDATES"
do
    if [[ ! -s "$file" ]]
    then
        echo "STOP: required input is missing or empty: $file"
        exit 1
    fi
done

echo "=== CROSS-SOURCE FEASIBILITY AUDIT ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - \
    "$MASTER" \
    "$PLAN" \
    "$GAPS" \
    "$V3_CANDIDATES" \
    "$IMAGE_ROOT" \
    "$CANDIDATES" \
    "$MAPPING" \
    "$UNMATCHED" \
    "$REPORT" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image


(
    master_path,
    plan_path,
    gaps_path,
    v3_candidates_path,
    image_root,
    candidates_path,
    mapping_path,
    unmatched_path,
    report_path,
) = map(Path, sys.argv[1:])


def require(frame, columns, name):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise SystemExit(
            f"STOP: {name} is missing columns: "
            + ", ".join(sorted(missing))
        )


def strings(frame, columns):
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str)
    return frame


def groups(frame):
    values = frame["split_group"].fillna("").astype(str)
    return values.where(
        values.ne(""),
        "record:" + frame["record_id"].fillna("").astype(str),
    )


def stable_hash(*values):
    text = "|".join(str(value) for value in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    frame.to_parquet(pending, index=False)
    pending.replace(path)


def write_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pending.replace(path)


master = pd.read_parquet(master_path)
plan = pd.read_parquet(plan_path)
targets = pd.read_parquet(gaps_path)
v3_candidates = pd.read_parquet(v3_candidates_path)

required = [
    "record_id",
    "source",
    "image_url",
    "image_license",
    "family",
    "genus",
    "species",
    "split",
    "split_group",
]

require(master, required, "master manifest")
require(plan, required, "training plan")
require(targets, required, "remaining targets")
require(v3_candidates, required, "V3 candidates")

for frame in (master, plan, targets, v3_candidates):
    strings(frame, required)

if len(plan) != 300_000:
    raise SystemExit(
        f"STOP: expected 300,000 plan rows; found {len(plan):,}"
    )

if len(targets) != 89:
    raise SystemExit(
        f"STOP: expected 89 remaining targets; found {len(targets):,}"
    )

if targets["record_id"].duplicated().any():
    raise SystemExit("STOP: duplicate target record IDs")

target_families = set(targets["family"]) - {""}

plan_ids = set(plan["record_id"])
plan_urls = set(plan.loc[plan["image_url"].ne(""), "image_url"])
plan_groups = set(groups(plan))

reserved_ids = set(v3_candidates["record_id"])
reserved_urls = set(
    v3_candidates.loc[v3_candidates["image_url"].ne(""), "image_url"]
)
reserved_groups = set(groups(v3_candidates))

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

pool = master[
    master["source"].eq("iNaturalist")
    & master["family"].isin(target_families)
    & master["image_url"].ne("")
    & master["image_license"].isin({"CC0", "CC-BY", "CC-BY-SA"})
    & eligible
].copy()

pool["_effective_group"] = groups(pool)

pool = pool[
    ~pool["record_id"].isin(plan_ids)
    & ~pool["record_id"].isin(reserved_ids)
    & ~pool["image_url"].isin(plan_urls)
    & ~pool["image_url"].isin(reserved_urls)
    & ~pool["_effective_group"].isin(plan_groups)
    & ~pool["_effective_group"].isin(reserved_groups)
].copy()

pool = pool.drop_duplicates("record_id").reset_index(drop=True)


def cached_path(record_id):
    return (
        image_root
        / "inaturalist"
        / record_id[:2]
        / f"{record_id}.jpg"
    )


def valid_cached(record_id):
    path = cached_path(record_id)

    if not path.is_file() or path.stat().st_size == 0:
        return False

    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


pool["_cached"] = pool["record_id"].map(valid_cached)

print(f"Remaining targets: {len(targets):,}")
print(f"Eligible unused iNaturalist candidates: {len(pool):,}")

targets["_target_order"] = targets["record_id"].map(
    lambda value: stable_hash("42", value)
)
targets = targets.sort_values("_target_order").reset_index(drop=True)

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
        unmatched_rows.append(target.drop(labels=["_target_order"]).to_dict())
        continue

    same_species = (
        target["species"] != ""
    ) & candidates["species"].eq(target["species"])

    same_genus = (
        target["genus"] != ""
    ) & candidates["genus"].eq(target["genus"])

    candidates["_match_score"] = 1
    candidates.loc[same_genus, "_match_score"] = 2
    candidates.loc[same_species, "_match_score"] = 3

    candidates["_same_original_split"] = candidates["split"].eq(
        target["split"]
    )

    candidates["_stable"] = candidates["record_id"].map(
        lambda value: stable_hash(
            "42",
            target["record_id"],
            value,
        )
    )

    candidates = candidates.sort_values(
        [
            "_match_score",
            "_same_original_split",
            "_cached",
            "_stable",
        ],
        ascending=[False, False, False, True],
    )

    chosen = candidates.iloc[0].copy()

    score = int(chosen["_match_score"])
    match_level = {
        3: "species",
        2: "genus",
        1: "family",
    }[score]

    original_split = str(chosen["split"])
    assigned_split = str(target["split"])
    effective_group = str(chosen["_effective_group"])
    was_cached = bool(chosen["_cached"])

    chosen["split"] = assigned_split

    if was_cached and "local_path" in chosen.index:
        chosen["local_path"] = str(cached_path(chosen["record_id"]))

    if "planning_group" in target.index:
        chosen["planning_group"] = target["planning_group"]

    chosen = chosen.drop(
        labels=[
            "_effective_group",
            "_cached",
            "_match_score",
            "_same_original_split",
            "_stable",
        ],
        errors="ignore",
    )

    selected_rows.append(chosen.to_dict())

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
        "target_split": assigned_split,
        "candidate_original_split": original_split,
        "assigned_split": assigned_split,
        "target_split_group": target["split_group"],
        "replacement_split_group": effective_group,
        "target_image_url": target["image_url"],
        "replacement_image_url": chosen["image_url"],
        "replacement_already_cached": was_cached,
    })

    used_ids.add(str(chosen["record_id"]))
    used_urls.add(str(chosen["image_url"]))
    used_groups.add(effective_group)

if selected_rows:
    selected = pd.DataFrame(selected_rows)
else:
    selected = master.iloc[:0].copy()

mapping_columns = [
    "target_record_id",
    "replacement_record_id",
    "target_source",
    "replacement_source",
    "family",
    "match_level",
    "target_genus",
    "replacement_genus",
    "target_species",
    "replacement_species",
    "target_split",
    "candidate_original_split",
    "assigned_split",
    "target_split_group",
    "replacement_split_group",
    "target_image_url",
    "replacement_image_url",
    "replacement_already_cached",
]

mapping = pd.DataFrame(mapping_rows, columns=mapping_columns)

if unmatched_rows:
    unmatched = pd.DataFrame(unmatched_rows)
else:
    unmatched = targets.drop(
        columns=["_target_order"],
        errors="ignore",
    ).iloc[:0].copy()

if selected["record_id"].duplicated().any():
    raise SystemExit("STOP: duplicate replacement record IDs")

if not mapping.empty:
    if mapping["replacement_image_url"].duplicated().any():
        raise SystemExit("STOP: duplicate replacement image URLs")

    if mapping["replacement_split_group"].duplicated().any():
        raise SystemExit("STOP: duplicate replacement groups")

    if set(mapping["replacement_split_group"]) & plan_groups:
        raise SystemExit(
            "STOP: replacement group is already present in the plan"
        )

    if not mapping["family"].ne("").all():
        raise SystemExit("STOP: a replacement has an empty family")

plan_source_counts = Counter(plan["source"])
mapped_count = len(mapping)

projected_sources = dict(plan_source_counts)

if mapped_count:
    projected_sources["GBIF"] = (
        projected_sources.get("GBIF", 0) - mapped_count
    )
    projected_sources["iNaturalist"] = (
        projected_sources.get("iNaturalist", 0) + mapped_count
    )

report = {
    "schema": 4,
    "policy": (
        "Replace the remaining inaccessible GBIF rows with unused "
        "iNaturalist records from the same family. Prefer identical "
        "species, then genus, then family; prefer the original split "
        "and an already cached image. Candidate record IDs, URLs and "
        "groups must be absent from the current plan and V3 replacement "
        "set. Assign the target split without falsifying source metadata."
    ),
    "target_rows": len(targets),
    "candidate_pool_rows": len(pool),
    "mapped_rows": len(mapping),
    "unmatched_rows": len(unmatched),
    "already_cached_replacements": int(
        mapping["replacement_already_cached"].sum()
    ) if not mapping.empty else 0,
    "downloads_needed": int(
        (~mapping["replacement_already_cached"]).sum()
    ) if not mapping.empty else 0,
    "match_levels": dict(Counter(mapping["match_level"])),
    "candidate_original_splits": dict(
        Counter(mapping["candidate_original_split"])
    ),
    "assigned_target_splits": dict(
        Counter(mapping["assigned_split"])
    ),
    "remaining_families": dict(
        Counter(unmatched["family"])
    ) if not unmatched.empty else {},
    "projected_plan_source_counts": {
        str(key): int(value)
        for key, value in projected_sources.items()
    },
    "family_preserved": True,
    "assigned_split_preserved": True,
    "candidate_groups_absent_from_current_plan": True,
    "training_plan_modified": False,
    "images_downloaded": False,
}

write_parquet(selected, candidates_path)
write_parquet(mapping, mapping_path)
write_parquet(unmatched, unmatched_path)
write_json(report, report_path)

print()
print(json.dumps(report, indent=2, ensure_ascii=False))
print()
print(f"Candidate manifest -> {candidates_path}")
print(f"Mapping            -> {mapping_path}")
print(f"Remaining gaps      -> {unmatched_path}")
print(f"Report              -> {report_path}")
PY

echo
echo "=== V4 OUTPUTS ==="

ls -lh \
    "$CANDIDATES" \
    "$MAPPING" \
    "$UNMATCHED" \
    "$REPORT"

echo
echo "Cross-source feasibility audit completed"
echo "No training-plan or image files were modified"
