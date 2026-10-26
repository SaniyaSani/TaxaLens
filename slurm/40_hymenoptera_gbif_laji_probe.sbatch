#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-laji-probe
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"
IMAGE_ROOT="${IMAGE_ROOT:-/scratch/$USER/taxalens/image_cache}"

DOWNLOAD_DIR="$DATA_ROOT/pipeline/image_download"
PLAN="$DOWNLOAD_DIR/gbif_plan.parquet"
MISSING="$DOWNLOAD_DIR/gbif_recovery_v1/gbif_missing.parquet"

PROBE_DIR="$DOWNLOAD_DIR/gbif_recovery_v1/laji_probe_v1"
SAMPLE="$PROBE_DIR/laji_probe_plan.parquet"
CACHED="$PROBE_DIR/laji_probe_cached.parquet"
REPORT="$PROBE_DIR/laji_probe_report.json"
JOB_TMP="$DATA_ROOT/tmp/gbif-laji-probe-${SLURM_JOB_ID}"

mkdir -p \
    "$PROBE_DIR" \
    "$IMAGE_ROOT" \
    "$JOB_TMP"

export TMPDIR="$JOB_TMP"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "GBIF plan: $PLAN"
echo "Missing records: $MISSING"
echo "Probe output: $PROBE_DIR"
echo

if [[ ! -s "$PLAN" ]]
then
    echo "STOP: GBIF plan is missing: $PLAN"
    exit 1
fi

if [[ ! -s "$MISSING" ]]
then
    echo "STOP: GBIF missing-record manifest is missing: $MISSING"
    exit 1
fi

echo "=== PREPARE DETERMINISTIC LAJI SAMPLE ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - "$PLAN" "$MISSING" "$SAMPLE" <<'PYSAMPLE'
import hashlib
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

plan_path = Path(sys.argv[1])
missing_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])

plan = pd.read_parquet(plan_path)
missing = pd.read_parquet(missing_path)

required = {"record_id", "image_url"}
for name, frame in (("plan", plan), ("missing", missing)):
    absent = required - set(frame.columns)
    if absent:
        raise SystemExit(
            f"STOP: {name} manifest is missing columns: "
            + ", ".join(sorted(absent))
        )

hosts = missing["image_url"].fillna("").astype(str).map(
    lambda value: urlparse(value).hostname or ""
)

laji_missing = missing.loc[
    hosts.eq("image.laji.fi"),
    ["record_id"],
].copy()

if len(laji_missing) < 12:
    raise SystemExit(
        f"STOP: expected at least 12 missing image.laji.fi rows, "
        f"found {len(laji_missing):,}"
    )

missing_ids = set(laji_missing["record_id"].astype(str))

candidates = plan[
    plan["record_id"].astype(str).isin(missing_ids)
].copy()

if len(candidates) != len(missing_ids):
    raise SystemExit(
        "STOP: not every missing image.laji.fi record was found "
        "in the original GBIF plan"
    )

candidates["_probe_key"] = candidates["record_id"].astype(str).map(
    lambda value: hashlib.sha256(
        f"gbif-laji-probe-v1:{value}".encode("utf-8")
    ).hexdigest()
)

sample = (
    candidates
    .sort_values(["_probe_key", "record_id"])
    .head(12)
    .drop(columns=["_probe_key"])
    .copy()
)

if len(sample) != 12:
    raise SystemExit(
        f"STOP: expected 12 probe rows, found {len(sample):,}"
    )

if sample["record_id"].astype(str).duplicated().any():
    raise SystemExit("STOP: duplicate record IDs in probe sample")

out_path.parent.mkdir(parents=True, exist_ok=True)
pending = out_path.with_name(out_path.name + ".pending")
sample.to_parquet(pending, index=False)
pending.replace(out_path)

print(f"Missing image.laji.fi rows: {len(laji_missing):,}")
print(f"Prepared deterministic probe: {len(sample):,} rows -> {out_path}")
print()
print(
    sample[
        ["record_id", "family", "image_url"]
    ].to_string(index=False)
)
PYSAMPLE

echo
echo "=== RUN CONTROLLED DOWNLOAD PROBE ==="
echo "12 records; one worker; 120-second read timeout; one retry"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    scripts/download_images.py \
    --manifest "$SAMPLE" \
    --out "$CACHED" \
    --image-root "$IMAGE_ROOT" \
    --max-images 0 \
    --allow-unbounded \
    --workers 1 \
    --chunksize 12 \
    --sleep 1.0 \
    --connect-timeout 30 \
    --read-timeout 120 \
    --retries 1 \
    --retry-backoff 5

echo
echo "=== SUMMARIZE PROBE ==="

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - "$CACHED" "$REPORT" <<'PYREPORT'
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

cached_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])

if not cached_path.is_file() or cached_path.stat().st_size == 0:
    raise SystemExit("STOP: probe cached manifest is missing or empty")

frame = pd.read_parquet(cached_path)

if len(frame) != 12:
    raise SystemExit(
        f"STOP: expected 12 probe rows, found {len(frame):,}"
    )

valid_mask = frame["local_path"].fillna("").astype(str).map(
    lambda value: bool(
        value
        and Path(value).is_file()
        and Path(value).stat().st_size > 0
    )
)

valid = int(valid_mask.sum())
failed = int((~valid_mask).sum())

reasons = Counter(
    frame.loc[
        ~valid_mask,
        "exclusion_reason",
    ].fillna("").astype(str)
)

report = {
    "schema": 1,
    "host": "image.laji.fi",
    "sample_rows": len(frame),
    "valid_images": valid,
    "failed_images": failed,
    "valid_fraction": valid / len(frame),
    "failure_reasons": dict(sorted(reasons.items())),
    "settings": {
        "workers": 1,
        "sleep_seconds": 1.0,
        "connect_timeout_seconds": 30,
        "read_timeout_seconds": 120,
        "retries": 1,
        "retry_backoff_seconds": 5,
    },
}

report_path.parent.mkdir(parents=True, exist_ok=True)
pending = report_path.with_name(report_path.name + ".pending")
pending.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
pending.replace(report_path)

print(json.dumps(report, indent=2, ensure_ascii=False))
PYREPORT

echo
echo "=== PROBE OUTPUTS ==="

ls -lh \
    "$SAMPLE" \
    "$CACHED" \
    "$REPORT"

echo
echo "GBIF image.laji.fi probe completed"
echo "No training-plan files were modified"
