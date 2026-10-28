#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-shared-encoders
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

SOURCE_ROOT="/scratch/$USER/taxalens/cache/huggingface/hub"

FINAL="/shares/hawlitschek.ieu.uzh/ssagut/TaxaLens/shared_backbones/dinov3_bioclip2_v1"
STAGING="${FINAL}.pending-${SLURM_JOB_ID}"

DINO_MODEL="models--facebook--dinov3-vitb16-pretrain-lvd1689m"
BIOCLIP_MODEL="models--imageomics--bioclip-2"

echo "Compute node: $(hostname)"
echo "Source cache: $SOURCE_ROOT"
echo "Permanent shared cache: $FINAL"
echo

if [[ -d "$FINAL" ]]
then
    echo "Permanent checkpoint already exists; verifying"

    test -s "$FINAL/checkpoint_report.json"
    test -s "$FINAL/SHA256SUMS.txt"

    (
        cd "$FINAL"
        sha256sum --quiet -c SHA256SUMS.txt
    )

    echo "Existing shared encoder checkpoint is valid"
    exit 0
fi

test -d "$SOURCE_ROOT/$DINO_MODEL"
test -d "$SOURCE_ROOT/$BIOCLIP_MODEL"

mkdir -p \
    "$STAGING/cache/huggingface/hub/$DINO_MODEL" \
    "$STAGING/cache/huggingface/hub/$BIOCLIP_MODEL"

cleanup() {
    if [[ -d "$STAGING" ]]
    then
        rm -rf -- "$STAGING"
    fi
}

trap cleanup EXIT

echo "=== COPY BOTH BACKBONES IN PARALLEL ==="

rsync -a \
    --info=stats2 \
    "$SOURCE_ROOT/$DINO_MODEL/" \
    "$STAGING/cache/huggingface/hub/$DINO_MODEL/" \
    > "$STAGING/dino_copy.log" 2>&1 &
dino_pid=$!

rsync -a \
    --info=stats2 \
    "$SOURCE_ROOT/$BIOCLIP_MODEL/" \
    "$STAGING/cache/huggingface/hub/$BIOCLIP_MODEL/" \
    > "$STAGING/bioclip_copy.log" 2>&1 &
bioclip_pid=$!

wait "$dino_pid"
wait "$bioclip_pid"

cat "$STAGING/dino_copy.log"
cat "$STAGING/bioclip_copy.log"

DINO_DEST="$STAGING/cache/huggingface/hub/$DINO_MODEL"
BIOCLIP_DEST="$STAGING/cache/huggingface/hub/$BIOCLIP_MODEL"

echo
echo "=== VERIFY COPIED MODEL FILES ==="

broken_link=$(
    find "$STAGING/cache/huggingface/hub" \
        -xtype l \
        -print \
        -quit
)

if [[ -n "$broken_link" ]]
then
    echo "STOP: broken cache symlink: $broken_link"
    exit 1
fi

test -n "$(
    find -L "$DINO_DEST" \
        -type f \
        -name preprocessor_config.json \
        -print \
        -quit
)"

test -n "$(
    find -L "$DINO_DEST" \
        -type f \
        \( -name '*.safetensors' -o -name '*.bin' \) \
        -print \
        -quit
)"

test -n "$(
    find -L "$BIOCLIP_DEST" \
        -type f \
        -name open_clip_model.safetensors \
        -print \
        -quit
)"

python - "$STAGING" "$DINO_MODEL" "$BIOCLIP_MODEL" <<'PYREPORT'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
models = list(sys.argv[2:])

def model_report(name):
    path = root / "cache" / "huggingface" / "hub" / name
    files = [
        item for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    ]
    snapshots = path / "snapshots"

    return {
        "cache_directory": name,
        "regular_files": len(files),
        "stored_bytes": sum(item.stat().st_size for item in files),
        "snapshots": sorted(
            item.name
            for item in snapshots.iterdir()
            if item.is_dir()
        ) if snapshots.is_dir() else [],
    }

report = {
    "schema": 1,
    "snapshot_id": "taxalens_shared_dinov3_bioclip2_v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Shared frozen image encoders for Diptera and Hymenoptera inference",
    "models": {
        "DINOv3-ViT-B/16": model_report(models[0]),
        "BioCLIP-2": model_report(models[1]),
    },
    "runtime_cache_root": str(root / "cache"),
}

(root / "checkpoint_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print(json.dumps(report, indent=2, ensure_ascii=False))
PYREPORT

echo
echo "=== CREATE AND VERIFY CHECKSUMS ==="

(
    cd "$STAGING"

    find cache \
        -type f \
        -print0 |
        LC_ALL=C sort -z |
        xargs -0 -r sha256sum \
        > SHA256SUMS.txt

    sha256sum --quiet -c SHA256SUMS.txt
)

echo "All encoder checksums verified"

echo
echo "=== FINALIZE ATOMICALLY ==="

mkdir -p "$(dirname "$FINAL")"
mv "$STAGING" "$FINAL"

trap - EXIT

(
    cd "$FINAL"
    sha256sum --quiet -c SHA256SUMS.txt
)

if [[ -n "$(
    find "$FINAL/cache/huggingface/hub" \
        -xtype l \
        -print \
        -quit
)" ]]
then
    echo "STOP: permanent cache contains a broken symlink"
    exit 1
fi

echo
echo "=== PERMANENT SHARED ENCODERS ==="

du -sh "$FINAL"
cat "$FINAL/checkpoint_report.json"

echo
echo "Shared Diptera/Hymenoptera encoder checkpoint completed"
echo "Scratch cache was not deleted"
