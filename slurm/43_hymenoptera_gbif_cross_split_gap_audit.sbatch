#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-gbif-cross-audit
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
PLAN="$DATA_ROOT/pipeline/image_download/gbif_plan.parquet"
AUDIT_DIR="$DATA_ROOT/pipeline/image_download/gbif_recovery_v1/laji_replacement_audit"

V2_CANDIDATES="$AUDIT_DIR/laji_replacement_candidates_v2.parquet"
V2_MAPPING="$AUDIT_DIR/laji_replacement_mapping_v2.parquet"
V2_UNMATCHED="$AUDIT_DIR/laji_unmatched_targets_v2.parquet"

V3_CANDIDATES="$AUDIT_DIR/laji_replacement_candidates_v3.parquet"
V3_MAPPING="$AUDIT_DIR/laji_replacement_mapping_v3.parquet"
V3_UNMATCHED="$AUDIT_DIR/laji_unmatched_targets_v3.parquet"
V3_REPORT="$AUDIT_DIR/laji_replacement_audit_v3.json"

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Remaining targets: $V2_UNMATCHED"
echo "Candidate groups already present in the plan remain excluded"
echo "No training-plan or image files will be modified"
echo

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - \
    "$MASTER" \
    "$PLAN" \
    "$V2_CANDIDATES" \
    "$V2_MAPPING" \
    "$V2_UNMATCHED" \
    "$IMAGE_ROOT/gbif" \
    "$V3_CANDIDATES" \
    "$V3_MAPPING" \
    "$V3_UNMATCHED" \
    "$V3_REPORT" <<'PYCROSS'
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

(
    master_path,
    plan_path,
    v2_candidates_path,
    v2_mapping_path,
    v2_unmatched_path,
    cache_root,
    v3_candidates_path,
    v3_mapping_path,
    v3_unmatched_path,
    v3_report_path,
) = map(Path, sys.argv[1:])

master = pd.read_parquet(master_path).fillna("")
plan = pd.read_parquet(plan_path).fillna("")
v2_candidates = pd.read_parquet(v2_candidates_path).fillna("")
v2_mapping = pd.read_parquet(v2_mapping_path).fillna("")
targets = pd.read_parquet(v2_unmatched_path).fillna("")

if len(v2_mapping) != 6_483:
    raise SystemExit(
        f"STOP: expected 6,483 v2 mappings, found {len(v2_mapping):,}"
    )

if len(targets) != 92:
    raise SystemExit(
        f"STOP: expected 92 remaining targets, found {len(targets):,}"
    )

required = {
    "record_id",
    "source",
    "image_url",
    "family",
    "genus",
    "species",
    "split",
    "split_group",
}

for name, frame in (
    ("master", master),
    ("plan", plan),
    ("v2 candidates", v2_candidates),
    ("targets", targets),
):
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(
            f"STOP: {name} is missing columns: "
            + ", ".join(sorted(missing))
        )

plan_ids = set(plan["record_id"].astype(str))
plan_urls = {
    value for value in plan["image_url"].astype(str) if value
}
plan_groups = {
    value for value in plan["split_group"].astype(str) if value
}

v2_ids = set(v2_candidates["record_id"].astype(str))
v2_urls = {
    value
    for value in v2_candidates["image_url"].astype(str)
    if value
}
v2_groups = {
    value
    for value in v2_candidates["split_group"].astype(str)
    if value
}

target_families = set(targets["family"].astype(str))

pool = master[
    master["source"].astype(str).eq("GBIF")
    & master["family"].astype(str).isin(target_families)
    & master["image_url"].astype(str).ne("")
].copy()

pool_hosts = pool["image_url"].astype(str).map(
    lambda value: urlparse(value).hostname or ""
)

pool = pool[
    ~pool["record_id"].astype(str).isin(plan_ids | v2_ids)
    & ~pool["image_url"].astype(str).isin(plan_urls | v2_urls)
    & ~pool["split_group"].astype(str).isin(
        plan_groups | v2_groups
    )
    & ~pool_hosts.eq("image.laji.fi")
].copy()

pool = pool.drop_duplicates("record_id", keep="first")
pool = pool.drop_duplicates("image_url", keep="first")

cached_paths = {}

if cache_root.is_dir():
    for image_path in cache_root.rglob("*.jpg"):
        try:
            if image_path.is_file() and image_path.stat().st_size > 0:
                cached_paths.setdefault(
                    image_path.stem,
                    str(image_path),
                )
        except OSError:
            continue

pool["_cached_path"] = (
    pool["record_id"].astype(str).map(cached_paths).fillna("")
)
pool["_cached"] = pool["_cached_path"].astype(str).ne("")
pool["_rank_key"] = pool["record_id"].astype(str).map(
    lambda value: hashlib.sha256(
        f"laji-cross-split-v3:{value}".encode("utf-8")
    ).hexdigest()
)

pool = pool.sort_values(
    ["_cached", "_rank_key", "record_id"],
    ascending=[False, True, True],
).reset_index(drop=True)

print(f"Remaining targets: {len(targets):,}")
print(f"Unused cross-split GBIF candidates: {len(pool):,}")

records = pool.to_dict(orient="records")

species_pool = defaultdict(list)
genus_pool = defaultdict(list)
family_pool = defaultdict(list)

for index, candidate in enumerate(records):
    family = str(candidate["family"])
    genus = str(candidate["genus"])
    species = str(candidate["species"])

    family_pool[family].append(index)

    if genus:
        genus_pool[(family, genus)].append(index)

    if species:
        species_pool[(family, species)].append(index)

targets["_target_key"] = targets["record_id"].astype(str).map(
    lambda value: hashlib.sha256(
        f"laji-cross-target-v3:{value}".encode("utf-8")
    ).hexdigest()
)
targets = targets.sort_values(
    ["_target_key", "record_id"]
).drop(columns=["_target_key"])

used_ids = set()
used_urls = set()
used_groups = set()

additional_candidates = []
additional_mapping = []
still_unmatched = []


def choose(indices):
    for index in indices:
        candidate = records[index]
        record_id = str(candidate["record_id"])
        image_url = str(candidate["image_url"])
        group = (
            str(candidate["split_group"])
            or f"record:{record_id}"
        )

        if (
            record_id in used_ids
            or image_url in used_urls
            or group in used_groups
        ):
            continue

        used_ids.add(record_id)
        used_urls.add(image_url)
        used_groups.add(group)
        return candidate

    return None


for target in targets.to_dict(orient="records"):
    target_split = str(target["split"])
    family = str(target["family"])
    genus = str(target["genus"])
    species = str(target["species"])

    replacement = None
    match_level = ""

    if species:
        replacement = choose(
            species_pool[(family, species)]
        )
        if replacement is not None:
            match_level = "species"

    if replacement is None and genus:
        replacement = choose(
            genus_pool[(family, genus)]
        )
        if replacement is not None:
            match_level = "genus"

    if replacement is None:
        replacement = choose(family_pool[family])
        if replacement is not None:
            match_level = "family"

    if replacement is None:
        still_unmatched.append(target)
        continue

    original_split = str(replacement["split"])

    candidate_row = {
        column: replacement.get(column, "")
        for column in master.columns
    }
    candidate_row["split"] = target_split
    additional_candidates.append(candidate_row)

    additional_mapping.append({
        "target_record_id": str(target["record_id"]),
        "replacement_record_id": str(replacement["record_id"]),
        "split": target_split,
        "family": family,
        "target_genus": genus,
        "replacement_genus": str(replacement["genus"]),
        "target_species": species,
        "replacement_species": str(replacement["species"]),
        "match_level": match_level,
        "replacement_host": (
            urlparse(str(replacement["image_url"])).hostname or ""
        ),
        "already_cached": bool(replacement["_cached"]),
        "cached_path": str(replacement["_cached_path"]),
        "existing_plan_group": False,
        "policy_stage": "cross_master_split_new_group",
        "original_master_split": original_split,
        "assigned_split": target_split,
    })

additional_candidates = pd.DataFrame(
    additional_candidates,
    columns=master.columns,
)
additional_mapping = pd.DataFrame(additional_mapping)
still_unmatched = pd.DataFrame(
    still_unmatched,
    columns=plan.columns,
)

v2_mapping = v2_mapping.copy()
v2_mapping["original_master_split"] = v2_mapping["split"]
v2_mapping["assigned_split"] = v2_mapping["split"]

combined_candidates = pd.concat(
    [v2_candidates, additional_candidates],
    ignore_index=True,
)
combined_mapping = pd.concat(
    [v2_mapping, additional_mapping],
    ignore_index=True,
)

if combined_candidates["record_id"].astype(str).duplicated().any():
    raise SystemExit("STOP: duplicate replacement record IDs")

if combined_mapping["target_record_id"].astype(str).duplicated().any():
    raise SystemExit("STOP: duplicate target mappings")

if (
    combined_mapping["replacement_record_id"]
    .astype(str)
    .duplicated()
    .any()
):
    raise SystemExit("STOP: reused replacement records")

for output_path, frame in (
    (v3_candidates_path, combined_candidates),
    (v3_mapping_path, combined_mapping),
    (v3_unmatched_path, still_unmatched),
):
    pending = output_path.with_name(output_path.name + ".pending")
    frame.to_parquet(pending, index=False)
    pending.replace(output_path)

report = {
    "schema": 3,
    "target_rows": 6_575,
    "v2_mappings": len(v2_mapping),
    "cross_split_gap_targets": len(targets),
    "cross_split_gap_mapped": len(additional_mapping),
    "mapped_rows": len(combined_mapping),
    "unmatched_rows": len(still_unmatched),
    "already_cached_replacements": int(
        combined_mapping["already_cached"].astype(bool).sum()
    ),
    "downloads_needed": int(
        len(combined_mapping)
        - combined_mapping["already_cached"].astype(bool).sum()
    ),
    "cross_master_split_replacements": len(additional_mapping),
    "match_levels": dict(sorted(Counter(
        combined_mapping["match_level"].astype(str)
    ).items())),
    "cross_split_match_levels": dict(sorted(Counter(
        additional_mapping["match_level"].astype(str)
    ).items())) if not additional_mapping.empty else {},
    "remaining_families": dict(sorted(Counter(
        still_unmatched["family"].astype(str)
    ).items())) if not still_unmatched.empty else {},
}

pending = v3_report_path.with_name(v3_report_path.name + ".pending")
pending.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
pending.replace(v3_report_path)

print(json.dumps(report, indent=2, ensure_ascii=False))
print()
print(f"Final candidates -> {v3_candidates_path}")
print(f"Final mapping    -> {v3_mapping_path}")
print(f"Remaining gaps   -> {v3_unmatched_path}")
print(f"Report           -> {v3_report_path}")
PYCROSS

echo
echo "=== V3 OUTPUTS ==="

ls -lh \
    "$V3_CANDIDATES" \
    "$V3_MAPPING" \
    "$V3_UNMATCHED" \
    "$V3_REPORT"

echo
echo "Cross-split replacement audit completed"
echo "No training-plan or image files were modified"
