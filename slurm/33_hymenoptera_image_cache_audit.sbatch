#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-cache-audit
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"
IMAGE_ROOT="${IMAGE_ROOT:-/scratch/$USER/taxalens/image_cache}"

PLAN="$DATA_ROOT/pipeline/planning/training_plan_v1/training_plan.parquet"
OUT_DIR="$DATA_ROOT/pipeline/planning/image_cache_audit"

mkdir -p "$OUT_DIR"

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Training plan: $PLAN"
echo "Shared image cache: $IMAGE_ROOT"
echo "No images will be downloaded or deleted"

apptainer exec "$CONTAINER" "$VENV/bin/python" \
    - "$PLAN" "$IMAGE_ROOT" "$OUT_DIR" <<'PY'
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from PIL import Image

from scripts.download_images import safe_source


plan = Path(sys.argv[1]).resolve()
image_root = Path(sys.argv[2]).resolve()
out_dir = Path(sys.argv[3]).resolve()

if not plan.is_file():
    raise SystemExit(f"STOP: training plan is missing: {plan}")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


frame = pd.read_parquet(plan).fillna("")

required = {
    "source",
    "record_id",
    "image_url",
    "local_path",
}
missing = required - set(frame.columns)

if missing:
    raise SystemExit(
        "STOP: training plan is missing columns: "
        + ", ".join(sorted(missing))
    )

frame = frame[
    ["source", "record_id", "image_url", "local_path"]
].copy()

sources = sorted(frame["source"].astype(str).unique())

stats = {
    source: {
        "source": source,
        "rows": 0,
        "current_local_valid": 0,
        "deterministic_cache_valid": 0,
        "available_now": 0,
        "invalid_existing": 0,
        "missing_bioscan_archive": 0,
        "missing_url_download": 0,
        "missing_without_location": 0,
        "available_bytes": 0,
    }
    for source in sources
}

record_ids = defaultdict(set)
urls = defaultdict(set)
url_rows = Counter()
host_counts = defaultdict(Counter)
path_results = {}


def inspect_image(path: Path) -> tuple[bool, int, bool]:
    key = str(path)

    if key in path_results:
        return path_results[key]

    if not path.is_file():
        result = (False, 0, False)
        path_results[key] = result
        return result

    size = path.stat().st_size

    if size <= 0:
        result = (False, size, True)
        path_results[key] = result
        return result

    try:
        with Image.open(path) as image:
            image.verify()
        result = (True, size, True)
    except Exception:
        result = (False, size, True)

    path_results[key] = result
    return result


for number, row in enumerate(frame.itertuples(index=False), start=1):
    source = str(row.source).strip()
    record_id = str(row.record_id).strip()
    image_url = str(row.image_url).strip()
    local_path = str(row.local_path).strip()

    source_stats = stats[source]
    source_stats["rows"] += 1

    if record_id:
        record_ids[source].add(record_id)

    if image_url:
        urls[source].add(image_url)
        url_rows[source] += 1
        host_counts[source][urlsplit(image_url).netloc.casefold()] += 1

    available = False
    selected_size = 0
    inspected_this_row = set()

    if local_path:
        current = Path(local_path).expanduser()
        valid, size, exists = inspect_image(current)
        inspected_this_row.add(str(current))

        if valid:
            source_stats["current_local_valid"] += 1
            available = True
            selected_size = size
        elif exists:
            source_stats["invalid_existing"] += 1

    if image_url and record_id:
        destination = (
            image_root
            / safe_source(source)
            / record_id[:2]
            / f"{record_id}.jpg"
        )

        valid, size, exists = inspect_image(destination)

        if valid:
            source_stats["deterministic_cache_valid"] += 1
            if not available:
                available = True
                selected_size = size
        elif exists and str(destination) not in inspected_this_row:
            source_stats["invalid_existing"] += 1

    if available:
        source_stats["available_now"] += 1
        source_stats["available_bytes"] += selected_size
    elif source == "BIOSCAN-5M":
        source_stats["missing_bioscan_archive"] += 1
    elif image_url:
        source_stats["missing_url_download"] += 1
    else:
        source_stats["missing_without_location"] += 1

    if number % 25_000 == 0:
        print(f"Audited image locations: {number:,}/{len(frame):,}", flush=True)


cache_reference = {}

for source in sources:
    directory = image_root / safe_source(source)
    files = 0
    total_bytes = 0

    if directory.is_dir():
        for candidate in directory.rglob("*.jpg"):
            if candidate.is_file():
                files += 1
                total_bytes += candidate.stat().st_size

    average_bytes = total_bytes / files if files else 0

    cache_reference[source] = {
        "directory": str(directory),
        "files": files,
        "bytes": total_bytes,
        "average_bytes": round(average_bytes),
    }

    source_stats = stats[source]
    source_stats["unique_record_ids"] = len(record_ids[source])
    source_stats["duplicate_record_id_excess"] = (
        source_stats["rows"] - len(record_ids[source])
    )
    source_stats["url_rows"] = url_rows[source]
    source_stats["unique_urls"] = len(urls[source])
    source_stats["duplicate_url_excess"] = (
        url_rows[source] - len(urls[source])
    )
    source_stats["estimated_missing_url_bytes"] = round(
        source_stats["missing_url_download"] * average_bytes
    )


report = {
    "schema": 1,
    "plan": str(plan),
    "plan_sha256": digest(plan),
    "rows": len(frame),
    "image_root": str(image_root),
    "network_requests_made": 0,
    "files_modified_in_cache": 0,
    "by_source": stats,
    "cache_reference": cache_reference,
    "url_hosts_by_source": {
        source: dict(host_counts[source].most_common())
        for source in sources
    },
    "totals": {
        "available_now": sum(
            value["available_now"] for value in stats.values()
        ),
        "missing_bioscan_archive": sum(
            value["missing_bioscan_archive"]
            for value in stats.values()
        ),
        "missing_url_download": sum(
            value["missing_url_download"]
            for value in stats.values()
        ),
        "missing_without_location": sum(
            value["missing_without_location"]
            for value in stats.values()
        ),
        "estimated_missing_url_bytes": sum(
            value["estimated_missing_url_bytes"]
            for value in stats.values()
        ),
    },
}

json_path = out_dir / "image_cache_audit.json"
pending_json = json_path.with_suffix(".json.pending")
pending_json.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
pending_json.replace(json_path)

tsv_path = out_dir / "image_cache_audit.tsv"
fields = [
    "source",
    "rows",
    "unique_record_ids",
    "url_rows",
    "unique_urls",
    "current_local_valid",
    "deterministic_cache_valid",
    "available_now",
    "invalid_existing",
    "missing_bioscan_archive",
    "missing_url_download",
    "missing_without_location",
    "estimated_missing_url_bytes",
]

pending_tsv = tsv_path.with_suffix(".tsv.pending")

with pending_tsv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=fields,
        delimiter="\t",
    )
    writer.writeheader()

    for source in sources:
        writer.writerow({
            field: stats[source].get(field, "")
            for field in fields
        })

pending_tsv.replace(tsv_path)

print()
print(json.dumps(report, indent=2, ensure_ascii=False))
print()
print(f"JSON report -> {json_path}")
print(f"TSV report  -> {tsv_path}")
PY

echo
echo "=== AUDIT FILES ==="

ls -lh \
    "$OUT_DIR/image_cache_audit.json" \
    "$OUT_DIR/image_cache_audit.tsv"

echo
echo "Hymenoptera image-cache audit completed"
echo "No images were downloaded or deleted"
