#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-plan-audit
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"

MASTER="$DATA_ROOT/pipeline/corpus/master_manifest.parquet"
OUT_DIR="$DATA_ROOT/pipeline/planning"
REPORT="$OUT_DIR/master_planning_audit.json"
FAMILIES="$OUT_DIR/family_by_source.tsv"
FAMILY_SPLITS="$OUT_DIR/family_source_split.tsv"

mkdir -p "$OUT_DIR"

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"
export PYARROW_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Master manifest: $MASTER"
echo "Output directory: $OUT_DIR"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - "$MASTER" "$REPORT" "$FAMILIES" "$FAMILY_SPLITS" <<'PYAUDIT'
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


master = Path(sys.argv[1])
report_path = Path(sys.argv[2])
families_path = Path(sys.argv[3])
family_splits_path = Path(sys.argv[4])

if not master.is_file():
    raise SystemExit(f"STOP: master manifest is missing: {master}")

available = set(pq.read_schema(master).names)

wanted = [
    "record_id",
    "source",
    "split",
    "source_split",
    "family",
    "subfamily",
    "tribe",
    "genus",
    "species",
    "identification_level",
    "basis_of_record",
    "label_quality",
    "dna_bin",
    "eligible_supervised",
    "split_group",
    "duplicate_group_id",
    "specimen_group_id",
    "image_url",
    "local_path",
]

columns = [column for column in wanted if column in available]
frame = pd.read_parquet(master, columns=columns)

for column in frame.columns:
    if frame[column].dtype == object:
        frame[column] = frame[column].fillna("").astype(str)

if "source" not in frame or "split" not in frame or "family" not in frame:
    raise SystemExit("STOP: master lacks source, split or family")

frame["source"] = frame["source"].astype(str).str.strip()
frame["split"] = frame["split"].astype(str).str.strip()
frame["family"] = (
    frame["family"]
    .astype(str)
    .str.strip()
    .replace("", "__unresolved__")
)

def value_counts(column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].value_counts(dropna=False).items()
    }

def cross_table(row: str, column: str) -> dict[str, dict[str, int]]:
    table = pd.crosstab(frame[row], frame[column])
    return {
        str(index): {
            str(name): int(table.loc[index, name])
            for name in table.columns
        }
        for index in table.index
    }

unknown_splits = sorted(
    set(frame["split"]) - {"train", "val", "test"}
)

group_column = next(
    (
        column
        for column in (
            "split_group",
            "duplicate_group_id",
            "specimen_group_id",
            "record_id",
        )
        if column in frame
    ),
    None,
)

if not group_column:
    raise SystemExit("STOP: no group identifier is available")

groups = frame[group_column].astype(str)

if groups.eq("").any() and "record_id" in frame:
    groups = groups.where(
        groups.ne(""),
        "record:" + frame["record_id"].astype(str),
    )

group_split_counts = (
    pd.DataFrame({
        "group": groups,
        "split": frame["split"],
    })
    .groupby("group", sort=False)["split"]
    .nunique()
)

leaking_groups = int(group_split_counts.gt(1).sum())

family_by_source = (
    frame.groupby(["family", "source"])
    .size()
    .unstack(fill_value=0)
)

family_by_source["total"] = family_by_source.sum(axis=1)
family_by_source = family_by_source.sort_values(
    "total",
    ascending=False,
)
family_by_source.to_csv(families_path, sep="\t")

family_source_split = (
    frame.groupby(["family", "source", "split"])
    .size()
    .rename("rows")
    .reset_index()
    .sort_values(
        ["family", "source", "split"],
        kind="stable",
    )
)
family_source_split.to_csv(
    family_splits_path,
    sep="\t",
    index=False,
)

nonempty_by_source = {}

for column in ("dna_bin", "image_url", "local_path"):
    if column not in frame:
        continue

    mask = frame[column].astype(str).str.strip().ne("")
    nonempty_by_source[column] = {
        str(key): int(value)
        for key, value in frame.loc[mask].groupby("source").size().items()
    }

evaluation = frame["split"].isin(["val", "test"])

report = {
    "schema": 1,
    "master": str(master.resolve()),
    "rows": int(len(frame)),
    "columns_read": columns,
    "split_counts": value_counts("split"),
    "source_counts": value_counts("source"),
    "source_by_split": cross_table("source", "split"),
    "identification_level_counts": value_counts("identification_level"),
    "source_by_identification_level": (
        cross_table("source", "identification_level")
        if "identification_level" in frame
        else {}
    ),
    "label_quality_counts": value_counts("label_quality"),
    "basis_of_record_counts": value_counts("basis_of_record"),
    "potential_evaluation_rows": int(evaluation.sum()),
    "potential_evaluation_by_source": {
        str(key): int(value)
        for key, value in frame.loc[evaluation].groupby("source").size().items()
    },
    "group_column": group_column,
    "groups": int(groups.nunique()),
    "leaking_groups": leaking_groups,
    "unknown_splits": unknown_splits,
    "unique_families": int(
        frame.loc[frame["family"] != "__unresolved__", "family"].nunique()
    ),
    "unique_genera": (
        int(frame["genus"].astype(str).str.strip().replace("", pd.NA).nunique())
        if "genus" in frame
        else 0
    ),
    "unique_species": (
        int(frame["species"].astype(str).str.strip().replace("", pd.NA).nunique())
        if "species" in frame
        else 0
    ),
    "nonempty_by_source": nonempty_by_source,
    "top_families": [
        {
            "family": str(family),
            **{
                str(column): int(row[column])
                for column in family_by_source.columns
            },
        }
        for family, row in family_by_source.head(40).iterrows()
    ],
}

if unknown_splits:
    raise SystemExit(
        "STOP: unknown dataset splits: " + ", ".join(unknown_splits)
    )

if leaking_groups:
    raise SystemExit(
        f"STOP: {leaking_groups} specimen groups cross dataset splits"
    )

pending = report_path.with_suffix(".json.pending")
pending.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
pending.replace(report_path)

print(json.dumps(report, indent=2, ensure_ascii=False))
print()
print(f"Family table -> {families_path}")
print(f"Family/source/split table -> {family_splits_path}")
PYAUDIT

echo
echo "=== AUDIT OUTPUTS ==="

ls -lh \
    "$REPORT" \
    "$FAMILIES" \
    "$FAMILY_SPLITS"

echo
echo "Hymenoptera planning audit completed"
