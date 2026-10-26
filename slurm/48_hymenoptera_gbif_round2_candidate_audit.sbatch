#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-round2-audit
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

DOWNLOAD_DIR="$DATA_ROOT/pipeline/image_download"
AUDIT_DIR="$DOWNLOAD_DIR/gbif_recovery_v1/laji_replacement_audit"
RECOVERY_FINAL="$DOWNLOAD_DIR/gbif_parallel_recovery_v1/final"

V3_CANDIDATES="$AUDIT_DIR/laji_replacement_candidates_v3.parquet"
V3_MAPPING="$AUDIT_DIR/laji_replacement_mapping_v3.parquet"

V4_CANDIDATES="$AUDIT_DIR/cross_source_v4/cross_source_candidates_v4.parquet"
V4_MAPPING="$AUDIT_DIR/cross_source_v4/cross_source_mapping_v4.parquet"
V4_UNMATCHED="$AUDIT_DIR/cross_source_v4/cross_source_unmatched_v4.parquet"

REPLACEMENT_ALL="$RECOVERY_FINAL/replacement_cached_merged.parquet"
REPLACEMENT_MISSING="$RECOVERY_FINAL/replacement_missing.parquet"
RETRY_ALL="$RECOVERY_FINAL/original_retry_cached_merged.parquet"
RETRY_MISSING="$RECOVERY_FINAL/original_retry_missing.parquet"

OUT_DIR="$DOWNLOAD_DIR/gbif_round2_candidates_v1"
CANDIDATES="$OUT_DIR/round2_candidates.parquet"
MAPPING="$OUT_DIR/round2_mapping.parquet"
UNMATCHED="$OUT_DIR/round2_unmatched_targets.parquet"
REPORT="$OUT_DIR/round2_candidate_audit.json"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "No images will be downloaded"
echo "No training-plan files will be modified"
echo

for file in \
    "$MASTER" \
    "$PLAN" \
    "$V3_CANDIDATES" \
    "$V3_MAPPING" \
    "$V4_CANDIDATES" \
    "$V4_MAPPING" \
    "$V4_UNMATCHED" \
    "$REPLACEMENT_ALL" \
    "$REPLACEMENT_MISSING" \
    "$RETRY_ALL" \
    "$RETRY_MISSING"
do
    if [[ ! -s "$file" ]]
    then
        echo "STOP: missing or empty input: $file"
        exit 1
    fi
done

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - \
    "$MASTER" \
    "$PLAN" \
    "$V3_CANDIDATES" \
    "$V3_MAPPING" \
    "$V4_CANDIDATES" \
    "$V4_MAPPING" \
    "$V4_UNMATCHED" \
    "$REPLACEMENT_ALL" \
    "$REPLACEMENT_MISSING" \
    "$RETRY_ALL" \
    "$RETRY_MISSING" \
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


paths = list(map(Path, sys.argv[1:]))

(
    master_path,
    plan_path,
    v3_candidates_path,
    v3_mapping_path,
    v4_candidates_path,
    v4_mapping_path,
    v4_unmatched_path,
    replacement_all_path,
    replacement_missing_path,
    retry_all_path,
    retry_missing_path,
    image_root,
    candidates_path,
    mapping_path,
    unmatched_path,
    report_path,
) = paths


STRING_COLUMNS = [
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
    "exclusion_reason",
]


def normalize(frame):
    for column in STRING_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str)
    return frame


def groups(frame):
    values = frame["split_group"].fillna("").astype(str)
    return values.where(
        values.ne(""),
        "record:" + frame["record_id"].astype(str),
    )


def hostname(value):
    return (urlparse(str(value)).hostname or "").lower()


def safe_source(value):
    return (
        re.sub(r"[^a-z0-9._-]+", "-", str(value).lower()).strip("-")
        or "unknown"
    )


def cache_path(row):
    record_id = str(row["record_id"])
    return (
        image_root
        / safe_source(row["source"])
        / record_id[:2]
        / f"{record_id}.jpg"
    )


def valid_local(value):
    if not value:
        return False

    path = Path(value)
    return path.is_file() and path.stat().st_size > 0


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

v3_candidates = normalize(pd.read_parquet(v3_candidates_path))
v3_mapping = pd.read_parquet(v3_mapping_path)

v4_candidates = normalize(pd.read_parquet(v4_candidates_path))
v4_mapping = pd.read_parquet(v4_mapping_path)
v4_unmatched = normalize(pd.read_parquet(v4_unmatched_path))

replacement_all = normalize(pd.read_parquet(replacement_all_path))
replacement_missing = normalize(pd.read_parquet(replacement_missing_path))

retry_all = normalize(pd.read_parquet(retry_all_path))
retry_missing = normalize(pd.read_parquet(retry_missing_path))

for frame, label in (
    (master, "master"),
    (plan, "plan"),
    (v3_candidates, "V3 candidates"),
    (v4_candidates, "V4 candidates"),
    (v4_unmatched, "V4 unmatched"),
):
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

    missing = required - set(frame.columns)

    if missing:
        raise SystemExit(
            f"STOP: {label} missing columns: "
            + ", ".join(sorted(missing))
        )

for mapping_frame, label in (
    (v3_mapping, "V3 mapping"),
    (v4_mapping, "V4 mapping"),
):
    required = {"target_record_id", "replacement_record_id"}
    missing = required - set(mapping_frame.columns)

    if missing:
        raise SystemExit(
            f"STOP: {label} missing columns: "
            + ", ".join(sorted(missing))
        )

all_mapping = pd.concat(
    [v3_mapping, v4_mapping],
    ignore_index=True,
    sort=False,
)

failed_replacement_ids = set(replacement_missing["record_id"])

failed_mappings = all_mapping[
    all_mapping["replacement_record_id"].astype(str).isin(
        failed_replacement_ids
    )
].copy()

if len(failed_mappings) != 2_840:
    raise SystemExit(
        "STOP: expected 2,840 failed replacement mappings; "
        f"found {len(failed_mappings):,}"
    )

laji_failed_targets = set(
    failed_mappings["target_record_id"].astype(str)
)

laji_unmapped_targets = set(
    v4_unmatched["record_id"].astype(str)
)

non_laji_failed_targets = set(
    retry_missing["record_id"].astype(str)
)

if len(laji_unmapped_targets) != 44:
    raise SystemExit(
        f"STOP: expected 44 unmapped Laji targets; "
        f"found {len(laji_unmapped_targets):,}"
    )

unresolved_ids = (
    laji_failed_targets
    | laji_unmapped_targets
    | non_laji_failed_targets
)

if len(unresolved_ids) != 4_587:
    raise SystemExit(
        f"STOP: expected 4,587 unresolved targets; "
        f"found {len(unresolved_ids):,}"
    )

target_origin = {}

for value in laji_failed_targets:
    target_origin[value] = "failed_replacement"

for value in laji_unmapped_targets:
    target_origin[value] = "unmapped_laji"

for value in non_laji_failed_targets:
    target_origin[value] = "failed_original_retry"

targets = plan[
    plan["record_id"].astype(str).isin(unresolved_ids)
].copy()

if len(targets) != 4_587:
    raise SystemExit(
        f"STOP: expected 4,587 target rows in plan; "
        f"found {len(targets):,}"
    )

targets["_origin"] = targets["record_id"].map(target_origin)

attempts = pd.concat(
    [replacement_all, retry_all],
    ignore_index=True,
    sort=False,
)

attempts["_host"] = attempts["image_url"].map(hostname)
attempts["_valid"] = attempts["local_path"].map(valid_local)

host_table = (
    attempts.groupby("_host", dropna=False)
    .agg(
        attempts=("record_id", "size"),
        valid=("_valid", "sum"),
    )
    .reset_index()
)

host_table["success_rate"] = (
    host_table["valid"] / host_table["attempts"]
)

host_rates = dict(
    zip(host_table["_host"], host_table["success_rate"])
)

bad_hosts = set(
    host_table.loc[
        host_table["attempts"].ge(5)
        & host_table["success_rate"].eq(0),
        "_host",
    ]
)

bad_hosts.add("image.laji.fi")

plan["_effective_group"] = groups(plan)
v3_candidates["_effective_group"] = groups(v3_candidates)
v4_candidates["_effective_group"] = groups(v4_candidates)

excluded_ids = (
    set(plan["record_id"])
    | set(v3_candidates["record_id"])
    | set(v4_candidates["record_id"])
)

excluded_urls = (
    set(plan.loc[plan["image_url"].ne(""), "image_url"])
    | set(
        v3_candidates.loc[
            v3_candidates["image_url"].ne(""),
            "image_url",
        ]
    )
    | set(
        v4_candidates.loc[
            v4_candidates["image_url"].ne(""),
            "image_url",
        ]
    )
)

excluded_groups = (
    set(plan["_effective_group"])
    | set(v3_candidates["_effective_group"])
    | set(v4_candidates["_effective_group"])
)

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

target_families = set(targets["family"]) - {""}

pool = master[
    master["source"].isin({"GBIF", "iNaturalist"})
    & master["family"].isin(target_families)
    & master["image_url"].ne("")
    & master["image_license"].isin({"CC0", "CC-BY", "CC-BY-SA"})
    & eligible
].copy()

pool["_effective_group"] = groups(pool)
pool["_host"] = pool["image_url"].map(hostname)

pool = pool[
    ~pool["record_id"].isin(excluded_ids)
    & ~pool["image_url"].isin(excluded_urls)
    & ~pool["_effective_group"].isin(excluded_groups)
    & ~pool["_host"].isin(bad_hosts)
].copy()

pool = pool.drop_duplicates("record_id").reset_index(drop=True)

pool["_cached"] = pool.apply(
    lambda row: (
        cache_path(row).is_file()
        and cache_path(row).stat().st_size > 0
    ),
    axis=1,
)

pool["_host_success_rate"] = (
    pool["_host"].map(host_rates).fillna(0.5)
)

family_options = Counter(pool["family"])

targets["_family_options"] = targets["family"].map(
    family_options
).fillna(0)

targets["_stable_order"] = targets["record_id"].map(
    lambda value: stable_hash("round2", "42", value)
)

targets = targets.sort_values(
    ["_family_options", "_stable_order"],
    ascending=[True, True],
).reset_index(drop=True)

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
            target.drop(
                labels=["_family_options", "_stable_order"],
                errors="ignore",
            ).to_dict()
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

    candidates["_same_source"] = candidates["source"].eq(
        target["source"]
    )

    candidates["_same_master_split"] = candidates["split"].eq(
        target["split"]
    )

    candidates["_stable"] = candidates["record_id"].map(
        lambda value: stable_hash(
            "round2",
            "42",
            target["record_id"],
            value,
        )
    )

    candidates = candidates.sort_values(
        [
            "_match_score",
            "_host_success_rate",
            "_cached",
            "_same_source",
            "_same_master_split",
            "_stable",
        ],
        ascending=[False, False, False, False, False, True],
    )

    chosen_for_target = candidates.head(3)

    for rank, (_, chosen) in enumerate(
        chosen_for_target.iterrows(),
        start=1,
    ):
        chosen = chosen.copy()

        score = int(chosen["_match_score"])
        match_level = {
            3: "species",
            2: "genus",
            1: "family",
        }[score]

        original_split = str(chosen["split"])
        chosen["split"] = str(target["split"])

        if bool(chosen["_cached"]) and "local_path" in chosen.index:
            chosen["local_path"] = str(cache_path(chosen))

        if "planning_group" in target.index:
            chosen["planning_group"] = target["planning_group"]

        clean = chosen.drop(
            labels=[
                "_effective_group",
                "_host",
                "_cached",
                "_host_success_rate",
                "_match_score",
                "_same_source",
                "_same_master_split",
                "_stable",
            ],
            errors="ignore",
        )

        selected_rows.append(clean.to_dict())

        mapping_rows.append({
            "target_record_id": target["record_id"],
            "target_origin": target["_origin"],
            "candidate_rank": rank,
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
            "candidate_original_split": original_split,
            "replacement_split_group": chosen["_effective_group"],
            "replacement_host": chosen["_host"],
            "observed_host_success_rate": float(
                chosen["_host_success_rate"]
            ),
            "replacement_already_cached": bool(chosen["_cached"]),
            "replacement_image_url": chosen["image_url"],
        })

        used_ids.add(str(chosen["record_id"]))
        used_urls.add(str(chosen["image_url"]))
        used_groups.add(str(chosen["_effective_group"]))

selected = (
    pd.DataFrame(selected_rows)
    if selected_rows
    else master.iloc[:0].copy()
)

mapping = pd.DataFrame(mapping_rows)

unmatched = (
    pd.DataFrame(unmatched_rows)
    if unmatched_rows
    else targets.iloc[:0].copy()
)

candidate_counts = (
    mapping.groupby("target_record_id").size()
    if not mapping.empty
    else pd.Series(dtype=int)
)

coverage = {
    "targets_with_1_candidate": int(candidate_counts.eq(1).sum()),
    "targets_with_2_candidates": int(candidate_counts.eq(2).sum()),
    "targets_with_3_candidates": int(candidate_counts.eq(3).sum()),
}

host_outcomes = {
    str(row["_host"]): {
        "attempts": int(row["attempts"]),
        "valid": int(row["valid"]),
        "success_rate": float(row["success_rate"]),
    }
    for _, row in host_table.sort_values(
        "attempts",
        ascending=False,
    ).iterrows()
}

report = {
    "schema": 1,
    "policy": (
        "For each unresolved training target, select up to three "
        "previously unused same-family alternatives. Prefer species, "
        "then genus, then family matches; prefer hosts with observed "
        "success, existing cached images, the original source and split. "
        "Current-plan and prior-replacement record IDs, URLs and groups "
        "remain excluded."
    ),
    "target_rows": len(targets),
    "target_origins": dict(Counter(targets["_origin"])),
    "candidate_pool_rows": len(pool),
    "candidate_rows": len(selected),
    "targets_with_candidates": int(candidate_counts.size),
    "targets_without_candidates": len(unmatched),
    **coverage,
    "candidate_sources": dict(Counter(mapping["replacement_source"]))
    if not mapping.empty else {},
    "candidate_hosts": dict(Counter(mapping["replacement_host"]))
    if not mapping.empty else {},
    "match_levels": dict(Counter(mapping["match_level"]))
    if not mapping.empty else {},
    "already_cached_candidates": int(
        mapping["replacement_already_cached"].sum()
    ) if not mapping.empty else 0,
    "remaining_families": dict(Counter(unmatched["family"]))
    if not unmatched.empty else {},
    "excluded_zero_success_hosts": sorted(bad_hosts),
    "observed_host_outcomes": host_outcomes,
    "training_plan_modified": False,
    "images_downloaded": False,
}

atomic_parquet(selected, candidates_path)
atomic_parquet(mapping, mapping_path)
atomic_parquet(unmatched, unmatched_path)
atomic_json(report, report_path)

print(json.dumps(report, indent=2, ensure_ascii=False))
print()
print(f"Candidate manifest -> {candidates_path}")
print(f"Candidate mapping  -> {mapping_path}")
print(f"Unmatched targets  -> {unmatched_path}")
print(f"Report             -> {report_path}")
PY

echo
echo "=== ROUND-2 AUDIT OUTPUTS ==="

ls -lh \
    "$CANDIDATES" \
    "$MAPPING" \
    "$UNMATCHED" \
    "$REPORT"

echo
echo "Round-2 candidate audit completed"
echo "No images were downloaded"
echo "No training-plan files were modified"
