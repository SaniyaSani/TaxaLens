#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-url-prepare
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"

PLAN="$DATA_ROOT/pipeline/planning/training_plan_gbif_recovered_v1/training_plan.parquet"
BASELINE="$DATA_ROOT/pipeline/url_baseline_v1"
MANIFEST="$BASELINE/training_manifest.parquet"
SHARDS="$BASELINE/manifest_shards"
REPORT="$BASELINE/preparation_report.json"

mkdir -p "$BASELINE" "$SHARDS"

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Source plan: $PLAN"
echo

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - "$PLAN" "$MANIFEST" "$REPORT" <<'PY'
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
report_path = Path(sys.argv[3])

frame = pd.read_parquet(source)

selected = frame[
    frame["source"].astype(str).isin(["GBIF", "iNaturalist"])
].copy()

if len(selected) != 144_817:
    raise SystemExit(
        f"STOP: expected 144,817 URL rows, found {len(selected):,}"
    )

if selected["record_id"].astype(str).duplicated().any():
    raise SystemExit("STOP: duplicate record IDs")

if selected["local_path"].fillna("").astype(str).eq("").any():
    raise SystemExit("STOP: empty local image path")

group_column = next(
    (
        name for name in (
            "split_group",
            "duplicate_group_id",
            "specimen_group_id",
            "record_id",
        )
        if name in selected
    ),
    None,
)

if group_column is None:
    raise SystemExit("STOP: no leakage-control group column")

groups = selected[group_column].fillna("").astype(str)

if "record_id" in selected:
    groups = groups.where(
        groups.ne(""),
        "record:" + selected["record_id"].astype(str),
    )

leaking = (
    pd.DataFrame({
        "group": groups,
        "split": selected["split"].astype(str),
    })
    .groupby("group")["split"]
    .nunique()
    .gt(1)
    .sum()
)

if leaking:
    raise SystemExit(f"STOP: {leaking} groups cross dataset splits")

paths = [Path(value) for value in selected["local_path"].astype(str)]

def valid(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False

workers = max(16, (os.cpu_count() or 8) * 4)

with ThreadPoolExecutor(max_workers=workers) as pool:
    validity = list(pool.map(valid, paths))

missing = len(validity) - sum(validity)

if missing:
    raise SystemExit(
        f"STOP: {missing:,} URL images are missing or empty"
    )

destination.parent.mkdir(parents=True, exist_ok=True)
pending = destination.with_name(destination.name + ".pending")
selected.to_parquet(pending, index=False)
pending.replace(destination)

def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

report = {
    "schema": 1,
    "input_plan": str(source),
    "input_plan_sha256": digest(source),
    "training_manifest": str(destination),
    "training_manifest_sha256": digest(destination),
    "rows": len(selected),
    "source_counts": {
        str(key): int(value)
        for key, value in selected["source"].value_counts().items()
    },
    "split_counts": {
        str(key): int(value)
        for key, value in selected["split"].value_counts().items()
    },
    "families": int(selected["family"].astype(str).nunique()),
    "genera": int(
        selected["genus"].astype(str).replace("", pd.NA).nunique()
    ),
    "species": int(
        selected["species"].astype(str).replace("", pd.NA).nunique()
    ),
    "group_column": group_column,
    "leaking_groups": int(leaking),
    "missing_images": missing,
}

temporary = report_path.with_suffix(".json.pending")
temporary.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
temporary.replace(report_path)

print(json.dumps(report, indent=2, ensure_ascii=False))
PY

echo
echo "=== CREATE GROUP-SAFE EMBEDDING SHARDS ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/make_embedding_shards.py \
    --input "$MANIFEST" \
    --out-dir "$SHARDS" \
    --shard-size 1000 \
    --seed 42

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - "$SHARDS/shards.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

if report["rows"] != 144_817:
    raise SystemExit("STOP: incorrect embedding-shard row count")

if report["shard_count"] != 145:
    raise SystemExit(
        f"STOP: expected 145 shards, found {report['shard_count']}"
    )

print("Prepared 145 group-safe GPU shards")
PY

echo
echo "Hymenoptera URL baseline preparation completed"
