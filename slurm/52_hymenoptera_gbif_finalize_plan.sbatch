#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-finalize-plan
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
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
INAT="$DATA_ROOT/pipeline/image_download/inaturalist_cached.parquet"

OUTPUT_PARENT="$DATA_ROOT/pipeline/planning"
OUTPUT="$OUTPUT_PARENT/training_plan_gbif_recovered_v1"
STAGING="$OUTPUT_PARENT/.training_plan_gbif_recovered_v1.pending-${SLURM_JOB_ID}"
export TAXALENS_FINAL_TRAINING_PLAN="$OUTPUT/training_plan.parquet"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Original plan: $PLAN"
echo "New plan:      $OUTPUT"
echo "Original plan will not be overwritten"
echo

if [[ -e "$OUTPUT" ]]
then
    echo "STOP: final output already exists: $OUTPUT"
    exit 1
fi

mkdir -p "$STAGING"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - "$PLAN" "$RECOVERY" "$PARALLEL" "$ROUND2" \
    "$INAT" "$STAGING" <<'PY'
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

plan_path = Path(sys.argv[1])
recovery = Path(sys.argv[2])
parallel = Path(sys.argv[3])
round2 = Path(sys.argv[4])
inat_path = Path(sys.argv[5])
output = Path(sys.argv[6])

paths = {
    "plan": plan_path,
    "initial_valid": recovery / "gbif_valid.parquet",
    "initial_missing": recovery / "gbif_missing.parquet",
    "retry_valid": parallel / "original_retry_valid.parquet",
    "round1_valid": parallel / "replacement_valid.parquet",
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
    "round2_replacements": (
        round2 / "round2_selected_replacements.parquet"
    ),
    "round2_unresolved": (
        round2 / "round2_unresolved_targets.parquet"
    ),
    "inaturalist_cached": inat_path,
}

for name, path in paths.items():
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"STOP: missing or empty {name}: {path}")

frames = {
    name: pd.read_parquet(path)
    for name, path in paths.items()
}

for frame in frames.values():
    if "record_id" in frame.columns:
        frame["record_id"] = frame["record_id"].fillna("").astype(str)

plan = frames["plan"]

if len(plan) != 300_000:
    raise SystemExit(
        f"STOP: expected 300,000 original rows, found {len(plan):,}"
    )

if plan["record_id"].eq("").any():
    raise SystemExit("STOP: original plan contains empty record IDs")

if plan["record_id"].duplicated().any():
    raise SystemExit("STOP: original plan contains duplicate record IDs")

plan_columns = list(plan.columns)
plan_index = plan.set_index("record_id", drop=False)
positions = {
    record_id: position
    for position, record_id in enumerate(plan["record_id"])
}

def pick(frame, names, label):
    for name in names:
        if name in frame.columns:
            return name
    raise SystemExit(
        f"STOP: cannot identify {label}; columns: "
        + ", ".join(map(str, frame.columns))
    )

def record_ids(frame):
    values = frame["record_id"].fillna("").astype(str)
    if values.eq("").any():
        raise SystemExit("STOP: empty record ID in input frame")
    if values.duplicated().any():
        raise SystemExit("STOP: duplicate record ID in input frame")
    return set(values)

def mapping_dictionary(frame):
    target_column = pick(
        frame,
        [
            "target_record_id",
            "original_record_id",
            "failed_record_id",
        ],
        "target-record column",
    )
    candidate_column = pick(
        frame,
        [
            "candidate_record_id",
            "replacement_record_id",
            "selected_record_id",
        ],
        "candidate-record column",
    )

    targets = frame[target_column].fillna("").astype(str)
    candidates = frame[candidate_column].fillna("").astype(str)

    if targets.eq("").any() or candidates.eq("").any():
        raise SystemExit("STOP: empty ID in replacement mapping")

    result = {}

    for target, candidate in zip(targets, candidates):
        previous = result.get(candidate)
        if previous is not None and previous != target:
            raise SystemExit(
                f"STOP: candidate {candidate} maps to multiple targets"
            )
        result[candidate] = target

    return result

def align_existing(frame, expected_ids, label):
    ids = frame["record_id"].fillna("").astype(str).tolist()

    if set(ids) != set(expected_ids):
        raise SystemExit(
            f"STOP: {label} record set differs from expected set"
        )

    target = plan_index.loc[ids].reset_index(drop=True)
    result = frame.reset_index(drop=True).copy()

    for column in plan_columns:
        if column not in result.columns:
            result[column] = target[column].to_numpy()

    for column in ("split", "planning_group"):
        result[column] = target[column].to_numpy()

    result["eligible_supervised"] = True
    result["exclusion_reason"] = ""
    result = result[plan_columns]
    result["_original_position"] = [
        positions[record_id] for record_id in ids
    ]

    return result

def prepare_replacements(frame, target_by_candidate, stage):
    result = frame.reset_index(drop=True).copy()
    candidate_ids = result["record_id"].fillna("").astype(str).tolist()

    if len(candidate_ids) != len(set(candidate_ids)):
        raise SystemExit(
            f"STOP: duplicate candidate IDs in {stage}"
        )

    missing_mapping = [
        value
        for value in candidate_ids
        if value not in target_by_candidate
    ]

    if missing_mapping:
        raise SystemExit(
            f"STOP: {stage} has {len(missing_mapping):,} "
            "candidates without target mappings"
        )

    target_ids = [
        target_by_candidate[value]
        for value in candidate_ids
    ]

    if len(target_ids) != len(set(target_ids)):
        raise SystemExit(
            f"STOP: multiple {stage} candidates selected "
            "for the same target"
        )

    target = plan_index.loc[target_ids].reset_index(drop=True)

    for column in plan_columns:
        if column not in result.columns:
            result[column] = target[column].to_numpy()

    candidate_families = (
        result["family"].fillna("").astype(str).to_numpy()
    )
    target_families = (
        target["family"].fillna("").astype(str).to_numpy()
    )

    family_mismatch = candidate_families != target_families

    if family_mismatch.any():
        raise SystemExit(
            f"STOP: {stage} contains "
            f"{int(family_mismatch.sum()):,} family mismatches"
        )

    original_candidate_split = (
        result["split"].fillna("").astype(str).tolist()
    )

    result["split"] = target["split"].to_numpy()
    result["planning_group"] = target["planning_group"].to_numpy()
    result["eligible_supervised"] = True
    result["exclusion_reason"] = ""

    ledger = pd.DataFrame({
        "stage": stage,
        "target_record_id": target_ids,
        "replacement_record_id": candidate_ids,
        "target_family": target_families,
        "replacement_family": candidate_families,
        "target_split": target["split"].fillna("").astype(str),
        "replacement_original_split": original_candidate_split,
        "replacement_source": result["source"].fillna("").astype(str),
        "replacement_local_path": result["local_path"].fillna("").astype(str),
    })

    result = result[plan_columns]
    result["_original_position"] = [
        positions[target_id]
        for target_id in target_ids
    ]

    return result, ledger, set(target_ids)

source_counts = {
    str(key): int(value)
    for key, value in plan["source"]
    .fillna("")
    .astype(str)
    .value_counts()
    .items()
}

expected_original_sources = {
    "BIOSCAN-5M": 154_115,
    "GBIF": 85_000,
    "iNaturalist": 60_885,
}

if source_counts != expected_original_sources:
    raise SystemExit(
        f"STOP: unexpected original source counts: {source_counts}"
    )

gbif_plan_ids = set(
    plan.loc[
        plan["source"].fillna("").astype(str).eq("GBIF"),
        "record_id",
    ]
)

inat_plan_ids = set(
    plan.loc[
        plan["source"].fillna("").astype(str).eq("iNaturalist"),
        "record_id",
    ]
)

initial_valid_ids = record_ids(frames["initial_valid"])
initial_missing_ids = record_ids(frames["initial_missing"])
retry_ids = record_ids(frames["retry_valid"])

if initial_valid_ids & initial_missing_ids:
    raise SystemExit("STOP: valid and missing GBIF sets overlap")

if initial_valid_ids | initial_missing_ids != gbif_plan_ids:
    raise SystemExit(
        "STOP: GBIF valid/missing sets do not partition the plan"
    )

if not retry_ids <= initial_missing_ids:
    raise SystemExit(
        "STOP: retry records occur outside the missing GBIF set"
    )

inat_ids = record_ids(frames["inaturalist_cached"])

if inat_ids != inat_plan_ids:
    raise SystemExit(
        "STOP: cached iNaturalist set differs from current plan"
    )

round1_mapping = {}

for mapping_name in ("v3_mapping", "v4_mapping"):
    for candidate, target in mapping_dictionary(
        frames[mapping_name]
    ).items():
        previous = round1_mapping.get(candidate)

        if previous is not None and previous != target:
            raise SystemExit(
                f"STOP: conflicting round-1 mapping for {candidate}"
            )

        round1_mapping[candidate] = target

round2_mapping = mapping_dictionary(frames["round2_mapping"])

bioscan = plan[
    plan["source"].fillna("").astype(str).eq("BIOSCAN-5M")
].copy()

bioscan["_original_position"] = [
    positions[value]
    for value in bioscan["record_id"].astype(str)
]

gbif_valid = align_existing(
    frames["initial_valid"],
    initial_valid_ids,
    "initial GBIF-valid",
)

inat_valid = align_existing(
    frames["inaturalist_cached"],
    inat_plan_ids,
    "iNaturalist cached",
)

retry_valid = align_existing(
    frames["retry_valid"],
    retry_ids,
    "original GBIF retry",
)

round1, ledger1, recovered1 = prepare_replacements(
    frames["round1_valid"],
    round1_mapping,
    "round1",
)

round2_rows, ledger2, recovered2 = prepare_replacements(
    frames["round2_replacements"],
    round2_mapping,
    "round2",
)

if retry_ids & recovered1:
    raise SystemExit("STOP: retry and round-1 targets overlap")

if retry_ids & recovered2:
    raise SystemExit("STOP: retry and round-2 targets overlap")

if recovered1 & recovered2:
    raise SystemExit("STOP: round-1 and round-2 targets overlap")

recovered_targets = retry_ids | recovered1 | recovered2
calculated_unresolved = initial_missing_ids - recovered_targets

unresolved_frame = frames["round2_unresolved"]
unresolved_column = pick(
    unresolved_frame,
    [
        "target_record_id",
        "original_record_id",
        "failed_record_id",
        "record_id",
    ],
    "unresolved target column",
)

reported_unresolved = set(
    unresolved_frame[unresolved_column]
    .fillna("")
    .astype(str)
)

if calculated_unresolved != reported_unresolved:
    raise SystemExit(
        "STOP: calculated and reported unresolved sets differ"
    )

parts = [
    bioscan,
    gbif_valid,
    inat_valid,
    retry_valid,
    round1,
    round2_rows,
]

final = pd.concat(parts, ignore_index=True, sort=False)
final = final.sort_values(
    "_original_position",
    kind="stable",
).drop(columns="_original_position").reset_index(drop=True)

# Every row in this materialized plan is supervised. Normalize these
# columns explicitly so mixed legacy bytes/bool values cannot corrupt
# Parquet type inference.
final["eligible_supervised"] = pd.Series(
    True,
    index=final.index,
    dtype="bool",
)
final["exclusion_reason"] = ""

if len(final) != 298_932:
    raise SystemExit(
        f"STOP: expected 298,932 final rows, found {len(final):,}"
    )

if final["record_id"].fillna("").astype(str).eq("").any():
    raise SystemExit("STOP: final plan contains empty record IDs")

if final["record_id"].astype(str).duplicated().any():
    raise SystemExit("STOP: final plan contains duplicate record IDs")

nonempty_urls = final["image_url"].fillna("").astype(str)
nonempty_urls = nonempty_urls[nonempty_urls.ne("")]

if nonempty_urls.duplicated().any():
    raise SystemExit("STOP: final plan contains duplicate image URLs")

final_source_counts = {
    str(key): int(value)
    for key, value in final["source"]
    .fillna("")
    .astype(str)
    .value_counts()
    .items()
}

expected_final_sources = {
    "BIOSCAN-5M": 154_115,
    "GBIF": 81_457,
    "iNaturalist": 63_360,
}

if final_source_counts != expected_final_sources:
    raise SystemExit(
        f"STOP: unexpected final source counts: {final_source_counts}"
    )

groups = final["split_group"].fillna("").astype(str)
groups = groups.where(
    groups.ne(""),
    "record:" + final["record_id"].astype(str),
)

leaking_groups = int(
    (
        pd.DataFrame({
            "group": groups,
            "split": final["split"].fillna("").astype(str),
        })
        .groupby("group")["split"]
        .nunique()
        > 1
    ).sum()
)

if leaking_groups:
    raise SystemExit(
        f"STOP: {leaking_groups:,} groups cross dataset splits"
    )

url_images = final[
    final["source"]
    .fillna("")
    .astype(str)
    .isin(["GBIF", "iNaturalist"])
].copy()

image_paths = (
    url_images["local_path"]
    .fillna("")
    .astype(str)
    .tolist()
)

def valid_image_path(value):
    if not value:
        return False

    path = Path(value)

    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False

file_check_workers = max(
    8,
    int(os.environ.get("SLURM_CPUS_PER_TASK", "4")) * 8,
)

with ThreadPoolExecutor(
    max_workers=file_check_workers
) as pool:
    file_validity = list(
        pool.map(valid_image_path, image_paths)
    )

missing_files = [
    value
    for value, valid in zip(image_paths, file_validity)
    if not valid
]

print(
    f"Validated {len(image_paths):,} URL-source image files "
    f"with {file_check_workers} parallel workers"
)

if missing_files:
    raise SystemExit(
        f"STOP: {len(missing_files):,} URL-source images "
        "are missing or empty"
    )

empty_hashes = int(
    url_images["image_sha256"]
    .fillna("")
    .astype(str)
    .eq("")
    .sum()
)

if empty_hashes:
    raise SystemExit(
        f"STOP: {empty_hashes:,} URL-source rows lack image hashes"
    )

if not final["eligible_supervised"].fillna(False).astype(bool).all():
    raise SystemExit(
        "STOP: final plan contains non-supervised rows"
    )

ledger = pd.concat([ledger1, ledger2], ignore_index=True)
dropped = plan_index.loc[
    sorted(calculated_unresolved)
].reset_index(drop=True).copy()

dropped["drop_reason"] = "image_unavailable_after_recovery"

output.mkdir(parents=True, exist_ok=True)

plan_out = output / "training_plan.parquet"
ledger_out = output / "replacement_ledger.parquet"
dropped_out = output / "dropped_targets.parquet"
report_out = output / "training_plan_report.json"

final.to_parquet(plan_out, index=False)
ledger.to_parquet(ledger_out, index=False)
dropped.to_parquet(dropped_out, index=False)

def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

report = {
    "schema": 1,
    "profile": "hymenoptera-teacher-gbif-recovered-v1",
    "input_plan": str(plan_path),
    "input_plan_sha256": digest(plan_path),
    "training_plan": os.environ["TAXALENS_FINAL_TRAINING_PLAN"],
    "training_plan_sha256": digest(plan_out),
    "rows": int(len(final)),
    "source_counts": final_source_counts,
    "split_counts": {
        str(key): int(value)
        for key, value in final["split"]
        .fillna("")
        .astype(str)
        .value_counts()
        .items()
    },
    "planning_group_counts": {
        str(key): int(value)
        for key, value in final["planning_group"]
        .fillna("")
        .astype(str)
        .value_counts()
        .items()
    },
    "unique_families": int(
        final["family"].fillna("").astype(str).nunique()
    ),
    "initial_gbif_missing": int(len(initial_missing_ids)),
    "original_retry_recovered": int(len(retry_ids)),
    "round1_recovered": int(len(recovered1)),
    "round2_recovered": int(len(recovered2)),
    "total_recovered": int(len(recovered_targets)),
    "dropped_unresolved": int(len(calculated_unresolved)),
    "url_images_ready": int(len(url_images)),
    "bioscan_images_pending": 154_115,
    "leaking_groups": leaking_groups,
    "all_url_images_present": True,
    "original_plan_preserved": True,
}

report_out.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

checksum_files = [
    plan_out,
    ledger_out,
    dropped_out,
    report_out,
]

checksum_text = "".join(
    f"{digest(path)}  {path.name}\n"
    for path in checksum_files
)

(output / "SHA256SUMS.txt").write_text(
    checksum_text,
    encoding="utf-8",
)

print(json.dumps(report, indent=2, ensure_ascii=False))
print()
print(f"Replacement ledger -> {ledger_out}")
print(f"Dropped targets    -> {dropped_out}")
print(f"Checksums          -> {output / 'SHA256SUMS.txt'}")
print("All final-plan validations passed")
PY

echo
echo "=== VERIFY STAGING CHECKSUMS ==="

(
    cd "$STAGING"
    sha256sum --quiet -c SHA256SUMS.txt
)

echo "All staging checksums verified"

echo
echo "=== FINALIZE ATOMICALLY ==="

mv "$STAGING" "$OUTPUT"

echo
echo "=== FINAL GBIF-RECOVERED PLAN ==="

ls -lh \
    "$OUTPUT/training_plan.parquet" \
    "$OUTPUT/training_plan_report.json" \
    "$OUTPUT/replacement_ledger.parquet" \
    "$OUTPUT/dropped_targets.parquet" \
    "$OUTPUT/SHA256SUMS.txt"

echo
du -sh "$OUTPUT"

echo
echo "Hymenoptera GBIF-recovered training plan completed"
echo "Original training_plan_v1 was not modified"
