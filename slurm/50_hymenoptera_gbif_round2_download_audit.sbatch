#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-round2-final
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"

CANDIDATE_DIR="$DATA_ROOT/pipeline/image_download/gbif_round2_candidates_v1"
DOWNLOAD_ROOT="$DATA_ROOT/pipeline/image_download/gbif_round2_download_v1"
CACHED_DIR="$DOWNLOAD_ROOT/cached"
FINAL_DIR="$DOWNLOAD_ROOT/final"

CANDIDATES="$CANDIDATE_DIR/round2_candidates.parquet"
MAPPING="$CANDIDATE_DIR/round2_mapping.parquet"
UNMATCHED="$CANDIDATE_DIR/round2_unmatched_targets.parquet"
CANDIDATE_REPORT="$CANDIDATE_DIR/round2_candidate_audit.json"

mkdir -p "$FINAL_DIR"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Candidate directory: $CANDIDATE_DIR"
echo "Cached shards: $CACHED_DIR"
echo "No training-plan files will be modified"
echo

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - \
    "$CANDIDATES" \
    "$MAPPING" \
    "$UNMATCHED" \
    "$CANDIDATE_REPORT" \
    "$CACHED_DIR" \
    "$FINAL_DIR" <<'PYAUDIT'
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

candidates_path = Path(sys.argv[1])
mapping_path = Path(sys.argv[2])
unmatched_path = Path(sys.argv[3])
candidate_report_path = Path(sys.argv[4])
cached_dir = Path(sys.argv[5])
final_dir = Path(sys.argv[6])

for path in (
    candidates_path,
    mapping_path,
    unmatched_path,
    candidate_report_path,
):
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"STOP: required input missing or empty: {path}")

candidate_report = json.loads(
    candidate_report_path.read_text(encoding="utf-8")
)

expected_candidates = int(candidate_report["candidate_rows"])
expected_targets = int(candidate_report["target_rows"])
expected_shards = 24

shard_paths = sorted(
    cached_dir.glob("round2_cached_part_*.parquet")
)

if len(shard_paths) != expected_shards:
    raise SystemExit(
        f"STOP: expected {expected_shards} cached shards, "
        f"found {len(shard_paths)}"
    )

print("=== LOAD COMPLETED SHARDS ===")

parts = []

for path in shard_paths:
    frame = pd.read_parquet(path)

    if frame.empty:
        raise SystemExit(f"STOP: empty cached shard: {path}")

    if "record_id" not in frame:
        raise SystemExit(
            f"STOP: shard has no record_id column: {path}"
        )

    print(f"{path.name}: {len(frame):,} rows")
    parts.append(frame)

cached = pd.concat(parts, ignore_index=True)
candidates = pd.read_parquet(candidates_path)
mapping = pd.read_parquet(mapping_path)
unmatched = pd.read_parquet(unmatched_path)

if len(cached) != expected_candidates:
    raise SystemExit(
        f"STOP: expected {expected_candidates:,} cached rows, "
        f"found {len(cached):,}"
    )

if len(candidates) != expected_candidates:
    raise SystemExit(
        f"STOP: expected {expected_candidates:,} source candidates, "
        f"found {len(candidates):,}"
    )

cached_ids = cached["record_id"].fillna("").astype(str)
candidate_ids = candidates["record_id"].fillna("").astype(str)

if cached_ids.eq("").any() or candidate_ids.eq("").any():
    raise SystemExit("STOP: empty candidate record_id")

if cached_ids.duplicated().any():
    raise SystemExit("STOP: duplicate record IDs in cached shards")

if candidate_ids.duplicated().any():
    raise SystemExit("STOP: duplicate source candidate record IDs")

if set(cached_ids) != set(candidate_ids):
    raise SystemExit(
        "STOP: cached record IDs do not match candidate manifest"
    )

target_column = next(
    (
        name
        for name in (
            "target_record_id",
            "original_record_id",
        )
        if name in mapping.columns
    ),
    None,
)

candidate_column = next(
    (
        name
        for name in (
            "replacement_record_id",
            "candidate_record_id",
        )
        if name in mapping.columns
    ),
    None,
)

rank_column = next(
    (
        name
        for name in (
            "candidate_rank",
            "replacement_rank",
            "rank",
        )
        if name in mapping.columns
    ),
    None,
)

if not target_column:
    raise SystemExit(
        "STOP: mapping has no target record identifier"
    )

if not candidate_column:
    raise SystemExit(
        "STOP: mapping has no candidate record identifier"
    )

mapping[target_column] = (
    mapping[target_column].fillna("").astype(str)
)
mapping[candidate_column] = (
    mapping[candidate_column].fillna("").astype(str)
)

if mapping[target_column].eq("").any():
    raise SystemExit("STOP: empty target ID in mapping")

if mapping[candidate_column].eq("").any():
    raise SystemExit("STOP: empty candidate ID in mapping")

if mapping[candidate_column].duplicated().any():
    raise SystemExit(
        "STOP: one candidate is assigned to multiple targets"
    )

if set(mapping[candidate_column]) != set(candidate_ids):
    raise SystemExit(
        "STOP: mapping candidate IDs do not match candidate manifest"
    )

if rank_column:
    mapping["_candidate_rank"] = pd.to_numeric(
        mapping[rank_column],
        errors="raise",
    ).astype(int)
else:
    mapping["_candidate_rank"] = (
        mapping.groupby(target_column, sort=False)
        .cumcount()
        .add(1)
    )

print()
print("=== CHECK LOCAL IMAGE FILES ===")

local_paths = cached["local_path"].fillna("").astype(str)

valid_mask = local_paths.map(
    lambda value: bool(value)
    and Path(value).is_file()
    and Path(value).stat().st_size > 0
)

valid = cached.loc[valid_mask].copy()
missing = cached.loc[~valid_mask].copy()

valid_ids = set(valid["record_id"].astype(str))

mapping["_valid_image"] = (
    mapping[candidate_column].isin(valid_ids)
)

ranked = mapping.sort_values(
    [target_column, "_candidate_rank", candidate_column],
    kind="mergesort",
)

selected_mapping = (
    ranked.loc[ranked["_valid_image"]]
    .groupby(target_column, sort=False)
    .head(1)
    .copy()
)

selected_standard = pd.DataFrame({
    "replacement_for_record_id":
        selected_mapping[target_column].astype(str),
    "replacement_record_id":
        selected_mapping[candidate_column].astype(str),
    "replacement_candidate_rank":
        selected_mapping["_candidate_rank"].astype(int),
})

for column in (
    "match_level",
    "target_origin",
    "target_family",
    "target_split",
):
    if column in selected_mapping:
        selected_standard[f"replacement_{column}"] = (
            selected_mapping[column].values
        )

selected_records = candidates.merge(
    selected_standard,
    left_on="record_id",
    right_on="replacement_record_id",
    how="inner",
    validate="one_to_one",
)

selected_records = selected_records.merge(
    valid[
        ["record_id", "local_path", "image_sha256"]
    ].rename(columns={
        "local_path": "_verified_local_path",
        "image_sha256": "_verified_image_sha256",
    }),
    on="record_id",
    how="inner",
    validate="one_to_one",
)

selected_records["local_path"] = (
    selected_records["_verified_local_path"]
)
selected_records["image_sha256"] = (
    selected_records["_verified_image_sha256"]
)
selected_records = selected_records.drop(
    columns=[
        "_verified_local_path",
        "_verified_image_sha256",
    ]
)

unmatched_id_column = next(
    (
        name
        for name in (
            "target_record_id",
            "original_record_id",
            "record_id",
        )
        if name in unmatched.columns
    ),
    None,
)

if not unmatched_id_column:
    raise SystemExit(
        "STOP: unmatched-target table has no record identifier"
    )

unmatched_ids = set(
    unmatched[unmatched_id_column]
    .fillna("")
    .astype(str)
)

unmatched_ids.discard("")

mapped_target_ids = set(
    mapping[target_column].astype(str)
)

all_target_ids = mapped_target_ids | unmatched_ids

if len(all_target_ids) != expected_targets:
    raise SystemExit(
        f"STOP: expected {expected_targets:,} unique targets, "
        f"found {len(all_target_ids):,}"
    )

selected_target_ids = set(
    selected_standard["replacement_for_record_id"]
    .astype(str)
)

unresolved_ids = sorted(
    all_target_ids - selected_target_ids
)

unresolved = pd.DataFrame({
    "target_record_id": unresolved_ids,
})

unresolved["unresolved_reason"] = (
    unresolved["target_record_id"].map(
        lambda value: (
            "no_round2_candidate"
            if value in unmatched_ids
            else "all_round2_candidates_failed"
        )
    )
)

def atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    pending = destination.with_name(
        destination.name + ".pending"
    )
    frame.to_parquet(pending, index=False)
    pending.replace(destination)

final_dir.mkdir(parents=True, exist_ok=True)

atomic_parquet(
    cached.sort_values("record_id"),
    final_dir / "round2_cached_merged.parquet",
)
atomic_parquet(
    valid.sort_values("record_id"),
    final_dir / "round2_valid_candidates.parquet",
)
atomic_parquet(
    missing.sort_values("record_id"),
    final_dir / "round2_missing_candidates.parquet",
)
atomic_parquet(
    selected_mapping.drop(
        columns=["_valid_image"],
        errors="ignore",
    ),
    final_dir / "round2_selected_mapping.parquet",
)
atomic_parquet(
    selected_records.sort_values(
        "replacement_for_record_id"
    ),
    final_dir / "round2_selected_replacements.parquet",
)
atomic_parquet(
    unresolved,
    final_dir / "round2_unresolved_targets.parquet",
)

failure_reasons = Counter(
    missing["exclusion_reason"]
    .fillna("")
    .astype(str)
)

selected_sources = Counter(
    selected_records["source"]
    .fillna("")
    .astype(str)
)

selected_ranks = Counter(
    selected_standard["replacement_candidate_rank"]
    .astype(int)
)

selected_hosts = Counter(
    selected_records["image_url"]
    .fillna("")
    .astype(str)
    .map(lambda value: urlparse(value).netloc)
)

match_level_column = next(
    (
        name
        for name in (
            "match_level",
            "replacement_match_level",
        )
        if name in selected_mapping.columns
    ),
    None,
)

if match_level_column:
    selected_match_levels = Counter(
        selected_mapping[match_level_column]
        .fillna("")
        .astype(str)
    )
else:
    selected_match_levels = Counter()

report = {
    "schema": 1,
    "candidate_rows": int(len(cached)),
    "valid_candidates": int(len(valid)),
    "missing_candidates": int(len(missing)),
    "candidate_valid_fraction": (
        float(len(valid) / len(cached))
        if len(cached)
        else 0.0
    ),
    "target_rows": int(expected_targets),
    "targets_with_candidates": int(
        candidate_report["targets_with_candidates"]
    ),
    "targets_without_candidates": int(
        candidate_report["targets_without_candidates"]
    ),
    "targets_recovered": int(len(selected_records)),
    "targets_still_unresolved": int(len(unresolved)),
    "target_recovery_fraction": float(
        len(selected_records) / expected_targets
    ),
    "selected_by_candidate_rank": {
        str(key): int(value)
        for key, value in sorted(selected_ranks.items())
    },
    "selected_by_source": {
        str(key): int(value)
        for key, value in sorted(selected_sources.items())
    },
    "selected_by_host": {
        str(key): int(value)
        for key, value in sorted(selected_hosts.items())
    },
    "selected_match_levels": {
        str(key): int(value)
        for key, value in sorted(
            selected_match_levels.items()
        )
    },
    "candidate_failure_reasons": {
        str(key): int(value)
        for key, value in sorted(failure_reasons.items())
    },
    "unresolved_by_reason": {
        str(key): int(value)
        for key, value in unresolved[
            "unresolved_reason"
        ].value_counts().items()
    },
    "projected_training_rows": int(
        300_000 - len(unresolved)
    ),
    "training_plan_modified": False,
    "images_modified_by_audit": False,
}

report_path = final_dir / "round2_download_audit.json"
pending_report = report_path.with_name(
    report_path.name + ".pending"
)
pending_report.write_text(
    json.dumps(report, indent=2, ensure_ascii=False)
    + "\n",
    encoding="utf-8",
)
pending_report.replace(report_path)

print(json.dumps(report, indent=2, ensure_ascii=False))
print()
print(
    "Selected replacements -> "
    f"{final_dir / 'round2_selected_replacements.parquet'}"
)
print(
    "Unresolved targets     -> "
    f"{final_dir / 'round2_unresolved_targets.parquet'}"
)
print(f"Report                 -> {report_path}")
PYAUDIT

echo
echo "=== ROUND-2 FINAL OUTPUTS ==="

find "$FINAL_DIR" \
    -maxdepth 1 \
    -type f \
    -printf '%12s bytes  %f\n' |
    sort

echo
echo "Round-2 download audit completed"
echo "No training-plan files were modified"
