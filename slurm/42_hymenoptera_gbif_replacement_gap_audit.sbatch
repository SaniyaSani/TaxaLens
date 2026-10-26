#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-gbif-gap-audit
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

V1_CANDIDATES="$AUDIT_DIR/laji_replacement_candidates.parquet"
V1_MAPPING="$AUDIT_DIR/laji_replacement_mapping.parquet"
V1_UNMATCHED="$AUDIT_DIR/laji_unmatched_targets.parquet"

V2_CANDIDATES="$AUDIT_DIR/laji_replacement_candidates_v2.parquet"
V2_MAPPING="$AUDIT_DIR/laji_replacement_mapping_v2.parquet"
V2_UNMATCHED="$AUDIT_DIR/laji_unmatched_targets_v2.parquet"
V2_REPORT="$AUDIT_DIR/laji_replacement_audit_v2.json"

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Gap targets: $V1_UNMATCHED"
echo "Existing plan groups may be reused only inside the same split"
echo "No training-plan or image files will be modified"
echo

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - \
    "$MASTER" \
    "$PLAN" \
    "$V1_CANDIDATES" \
    "$V1_MAPPING" \
    "$V1_UNMATCHED" \
    "$IMAGE_ROOT/gbif" \
    "$V2_CANDIDATES" \
    "$V2_MAPPING" \
    "$V2_UNMATCHED" \
    "$V2_REPORT" <<'PYGAP'
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
    v1_candidates_path,
    v1_mapping_path,
    v1_unmatched_path,
    cache_root,
    v2_candidates_path,
    v2_mapping_path,
    v2_unmatched_path,
    v2_report_path,
) = map(Path, sys.argv[1:])

master = pd.read_parquet(master_path).fillna("")
plan = pd.read_parquet(plan_path).fillna("")
v1_candidates = pd.read_parquet(v1_candidates_path).fillna("")
v1_mapping = pd.read_parquet(v1_mapping_path).fillna("")
targets = pd.read_parquet(v1_unmatched_path).fillna("")

if len(v1_mapping) != 6_245:
    raise SystemExit(
        f"STOP: expected 6,245 v1 mappings, found {len(v1_mapping):,}"
    )

if len(targets) != 330:
    raise SystemExit(
        f"STOP: expected 330 gap targets, found {len(targets):,}"
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
    ("v1 candidates", v1_candidates),
    ("gap targets", targets),
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

v1_ids = set(v1_candidates["record_id"].astype(str))
v1_urls = {
    value
    for value in v1_candidates["image_url"].astype(str)
    if value
}
v1_groups = {
    value
    for value in v1_candidates["split_group"].astype(str)
    if value
}

target_families = set(targets["family"].astype(str))
target_splits = set(targets["split"].astype(str))

pool = master[
    master["source"].astype(str).eq("GBIF")
    & master["family"].astype(str).isin(target_families)
    & master["split"].astype(str).isin(target_splits)
    & master["image_url"].astype(str).ne("")
].copy()

pool_hosts = pool["image_url"].astype(str).map(
    lambda value: urlparse(value).hostname or ""
)

pool = pool[
    ~pool["record_id"].astype(str).isin(plan_ids | v1_ids)
    & ~pool["image_url"].astype(str).isin(plan_urls | v1_urls)
    & ~pool["split_group"].astype(str).isin(v1_groups)
    & ~pool_hosts.eq("image.laji.fi")
].copy()

pool = pool.drop_duplicates("record_id", keep="first")
pool = pool.drop_duplicates("image_url", keep="first")

print(f"Gap targets: {len(targets):,}")
print(f"Relaxed candidate pool: {len(pool):,}")

cached_ids = set()

if cache_root.is_dir():
    for image_path in cache_root.rglob("*.jpg"):
        try:
            if image_path.is_file() and image_path.stat().st_size > 0:
                cached_ids.add(image_path.stem)
        except OSError:
            continue

pool["_cached"] = pool["record_id"].astype(str).isin(cached_ids)
pool["_existing_plan_group"] = (
    pool["split_group"].astype(str).isin(plan_groups)
)
pool["_rank_key"] = pool["record_id"].astype(str).map(
    lambda value: hashlib.sha256(
        f"laji-gap-v2:{value}".encode("utf-8")
    ).hexdigest()
)

pool = pool.sort_values(
    [
        "_cached",
        "_existing_plan_group",
        "_rank_key",
        "record_id",
    ],
    ascending=[False, True, True, True],
).reset_index(drop=True)

records = pool.to_dict(orient="records")

species_pool = defaultdict(list)
genus_pool = defaultdict(list)
family_pool = defaultdict(list)

for index, record in enumerate(records):
    split = str(record["split"])
    family = str(record["family"])
    genus = str(record["genus"])
    species = str(record["species"])

    family_pool[(split, family)].append(index)

    if genus:
        genus_pool[(split, family, genus)].append(index)

    if species:
        species_pool[(split, family, species)].append(index)

targets["_target_key"] = targets["record_id"].astype(str).map(
    lambda value: hashlib.sha256(
        f"laji-gap-target-v2:{value}".encode("utf-8")
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
    split = str(target["split"])
    family = str(target["family"])
    genus = str(target["genus"])
    species = str(target["species"])

    replacement = None
    match_level = ""

    if species:
        replacement = choose(
            species_pool[(split, family, species)]
        )
        if replacement is not None:
            match_level = "species"

    if replacement is None and genus:
        replacement = choose(
            genus_pool[(split, family, genus)]
        )
        if replacement is not None:
            match_level = "genus"

    if replacement is None:
        replacement = choose(
            family_pool[(split, family)]
        )
        if replacement is not None:
            match_level = "family"

    if replacement is None:
        still_unmatched.append(target)
        continue

    additional_candidates.append({
        column: replacement.get(column, "")
        for column in master.columns
    })

    additional_mapping.append({
        "target_record_id": str(target["record_id"]),
        "replacement_record_id": str(replacement["record_id"]),
        "split": split,
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
        "cached_path": "",
        "existing_plan_group": bool(
            replacement["_existing_plan_group"]
        ),
        "policy_stage": "same_split_relaxed_plan_group",
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

v1_mapping = v1_mapping.copy()
v1_mapping["existing_plan_group"] = False
v1_mapping["policy_stage"] = "strict_new_group"

combined_candidates = pd.concat(
    [v1_candidates, additional_candidates],
    ignore_index=True,
)
combined_mapping = pd.concat(
    [v1_mapping, additional_mapping],
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
    (v2_candidates_path, combined_candidates),
    (v2_mapping_path, combined_mapping),
    (v2_unmatched_path, still_unmatched),
):
    pending = output_path.with_name(output_path.name + ".pending")
    frame.to_parquet(pending, index=False)
    pending.replace(output_path)

report = {
    "schema": 2,
    "target_rows": 6_575,
    "strict_v1_mappings": len(v1_mapping),
    "gap_targets": len(targets),
    "gap_mapped": len(additional_mapping),
    "mapped_rows": len(combined_mapping),
    "unmatched_rows": len(still_unmatched),
    "already_cached_replacements": int(
        combined_mapping["already_cached"].astype(bool).sum()
    ),
    "downloads_needed": int(
        len(combined_mapping)
        - combined_mapping["already_cached"].astype(bool).sum()
    ),
    "existing_plan_group_replacements": int(
        combined_mapping["existing_plan_group"].astype(bool).sum()
    ),
    "match_levels": dict(sorted(Counter(
        combined_mapping["match_level"].astype(str)
    ).items())),
    "gap_match_levels": dict(sorted(Counter(
        additional_mapping["match_level"].astype(str)
    ).items())) if not additional_mapping.empty else {},
    "gap_families_remaining": dict(sorted(Counter(
        still_unmatched["family"].astype(str)
    ).items())) if not still_unmatched.empty else {},
}

pending = v2_report_path.with_name(v2_report_path.name + ".pending")
pending.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
pending.replace(v2_report_path)

print(json.dumps(report, indent=2, ensure_ascii=False))
print()
print(f"Combined candidates -> {v2_candidates_path}")
print(f"Combined mapping    -> {v2_mapping_path}")
print(f"Remaining gaps      -> {v2_unmatched_path}")
print(f"Report              -> {v2_report_path}")
PYGAP

echo
echo "=== GAP-AUDIT OUTPUTS ==="

ls -lh \
    "$V2_CANDIDATES" \
    "$V2_MAPPING" \
    "$V2_UNMATCHED" \
    "$V2_REPORT"

echo
echo "GBIF replacement gap audit completed"
echo "No training-plan or image files were modified"
