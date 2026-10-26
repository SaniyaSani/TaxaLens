#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-gbif-repl-audit
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
MISSING="$DATA_ROOT/pipeline/image_download/gbif_recovery_v1/gbif_missing.parquet"

OUT_DIR="$DATA_ROOT/pipeline/image_download/gbif_recovery_v1/laji_replacement_audit"
CANDIDATES="$OUT_DIR/laji_replacement_candidates.parquet"
MAPPING="$OUT_DIR/laji_replacement_mapping.parquet"
UNMATCHED="$OUT_DIR/laji_unmatched_targets.parquet"
REPORT="$OUT_DIR/laji_replacement_audit.json"

mkdir -p "$OUT_DIR"

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Master manifest: $MASTER"
echo "Current GBIF plan: $PLAN"
echo "Missing manifest: $MISSING"
echo "No plan or cached image files will be modified"
echo

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - \
    "$MASTER" \
    "$PLAN" \
    "$MISSING" \
    "$IMAGE_ROOT/gbif" \
    "$CANDIDATES" \
    "$MAPPING" \
    "$UNMATCHED" \
    "$REPORT" <<'PYAUDIT'
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
    missing_path,
    cache_root,
    candidates_path,
    mapping_path,
    unmatched_path,
    report_path,
) = map(Path, sys.argv[1:])

master = pd.read_parquet(master_path).fillna("")
plan = pd.read_parquet(plan_path).fillna("")
missing = pd.read_parquet(missing_path).fillna("")

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
    ("missing", missing),
):
    absent = required - set(frame.columns)
    if absent:
        raise SystemExit(
            f"STOP: {name} is missing columns: "
            + ", ".join(sorted(absent))
        )

missing_hosts = missing["image_url"].astype(str).map(
    lambda value: urlparse(value).hostname or ""
)

laji_missing = missing[missing_hosts.eq("image.laji.fi")].copy()
target_ids = set(laji_missing["record_id"].astype(str))

targets = plan[
    plan["record_id"].astype(str).isin(target_ids)
].copy()

if len(targets) != 6_575:
    raise SystemExit(
        f"STOP: expected 6,575 Laji targets, found {len(targets):,}"
    )

plan_ids = set(plan["record_id"].astype(str))
plan_urls = {
    value
    for value in plan["image_url"].astype(str)
    if value
}
plan_groups = {
    value
    for value in plan["split_group"].astype(str)
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
    ~pool["record_id"].astype(str).isin(plan_ids)
    & ~pool["image_url"].astype(str).isin(plan_urls)
    & ~pool["split_group"].astype(str).isin(plan_groups)
    & ~pool_hosts.eq("image.laji.fi")
].copy()

pool = pool.drop_duplicates("record_id", keep="first")
pool = pool.drop_duplicates("image_url", keep="first")

print(f"Laji targets: {len(targets):,}")
print(f"Eligible unused GBIF candidate pool: {len(pool):,}")

print()
print("=== INDEX EXISTING GBIF CACHE ===")

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

print(f"Non-empty cached GBIF files indexed: {len(cached_paths):,}")

pool["_cached_path"] = (
    pool["record_id"].astype(str).map(cached_paths).fillna("")
)
pool["_cached"] = pool["_cached_path"].astype(str).ne("")
pool["_rank_key"] = pool["record_id"].astype(str).map(
    lambda value: hashlib.sha256(
        f"laji-replacement-v1:{value}".encode("utf-8")
    ).hexdigest()
)

pool = pool.sort_values(
    ["_cached", "_rank_key", "record_id"],
    ascending=[False, True, True],
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
        f"laji-target-v1:{value}".encode("utf-8")
    ).hexdigest()
)

targets = targets.sort_values(
    ["_target_key", "record_id"]
).drop(columns=["_target_key"])

used_ids = set()
used_urls = set()
used_groups = set()

selected_rows = []
mapping_rows = []
unmatched_rows = []


def choose(indices):
    for index in indices:
        record = records[index]

        record_id = str(record["record_id"])
        image_url = str(record["image_url"])
        group = str(record["split_group"]) or f"record:{record_id}"

        if (
            record_id in used_ids
            or image_url in used_urls
            or group in used_groups
        ):
            continue

        used_ids.add(record_id)
        used_urls.add(image_url)
        used_groups.add(group)
        return record

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
        unmatched_rows.append(target)
        continue

    selected_rows.append({
        column: replacement.get(column, "")
        for column in master.columns
    })

    mapping_rows.append({
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
        "cached_path": str(replacement["_cached_path"]),
    })

selected = pd.DataFrame(selected_rows, columns=master.columns)
mapping = pd.DataFrame(mapping_rows)
unmatched = pd.DataFrame(unmatched_rows, columns=plan.columns)

if not mapping.empty:
    if mapping["replacement_record_id"].duplicated().any():
        raise SystemExit("STOP: duplicate replacement record IDs")

    if len(set(mapping["replacement_record_id"])) != len(mapping):
        raise SystemExit("STOP: replacement reuse detected")

    if not (
        mapping["target_record_id"].nunique()
        == len(mapping)
    ):
        raise SystemExit("STOP: duplicate target mappings")

for output_path, frame in (
    (candidates_path, selected),
    (mapping_path, mapping),
    (unmatched_path, unmatched),
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending = output_path.with_name(output_path.name + ".pending")
    frame.to_parquet(pending, index=False)
    pending.replace(output_path)

match_counts = (
    Counter(mapping["match_level"].astype(str))
    if not mapping.empty
    else Counter()
)

host_counts = (
    Counter(mapping["replacement_host"].astype(str))
    if not mapping.empty
    else Counter()
)

already_cached = (
    int(mapping["already_cached"].sum())
    if not mapping.empty
    else 0
)

report = {
    "schema": 1,
    "policy": (
        "Replace inaccessible image.laji.fi GBIF rows with unused "
        "GBIF rows from the same split and family; prefer identical "
        "species, then genus, then family; exclude current-plan "
        "record IDs, URLs and split groups; deterministic SHA-256 "
        "ordering; prefer already cached candidates."
    ),
    "target_rows": len(targets),
    "candidate_pool_rows": len(pool),
    "mapped_rows": len(mapping),
    "unmatched_rows": len(unmatched),
    "already_cached_replacements": already_cached,
    "replacement_downloads_needed": len(mapping) - already_cached,
    "match_levels": dict(sorted(match_counts.items())),
    "replacement_hosts": dict(
        sorted(
            host_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ),
}

report_path.parent.mkdir(parents=True, exist_ok=True)
pending = report_path.with_name(report_path.name + ".pending")
pending.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
pending.replace(report_path)

print()
print(json.dumps(report, indent=2, ensure_ascii=False))
print()
print(f"Candidate manifest -> {candidates_path}")
print(f"Mapping             -> {mapping_path}")
print(f"Unmatched targets   -> {unmatched_path}")
print(f"Audit report        -> {report_path}")
PYAUDIT

echo
echo "=== AUDIT OUTPUTS ==="

ls -lh \
    "$CANDIDATES" \
    "$MAPPING" \
    "$UNMATCHED" \
    "$REPORT"

echo
echo "GBIF Laji replacement feasibility audit completed"
echo "No training-plan or cached image files were modified"
