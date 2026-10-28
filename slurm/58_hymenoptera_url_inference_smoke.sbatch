#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-hym-infer-smoke
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=lowprio
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=L4:1
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"

CACHE_ROOT="${TAXALENS_CACHE_ROOT:-/scratch/$USER/taxalens/cache}"

export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT"
export TOKENIZERS_PARALLELISM=false

echo "Encoder cache: $HF_HOME"

CHECKPOINT="/shares/hawlitschek.ieu.uzh/$USER/TaxaLens/v11_hymenoptera_teacher/hymenoptera_url_baseline_v1"
MODEL_DIR="$CHECKPOINT/models"
MANIFEST="$CHECKPOINT/pipeline/training_manifest.parquet"

OUTPUT_DIR="/scratch/$USER/taxalens/v11_hymenoptera_teacher/pipeline/url_inference_smoke_v1"
REPORT="$OUTPUT_DIR/inference_smoke_report.json"
JOB_TMP="$OUTPUT_DIR/tmp-${SLURM_JOB_ID}"

mkdir -p "$OUTPUT_DIR" "$JOB_TMP"

export TMPDIR="$JOB_TMP"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Model directory: $MODEL_DIR"
echo "Test manifest:  $MANIFEST"
echo

for required in \
    "$MODEL_DIR/classifiers.joblib" \
    "$MODEL_DIR/retrieval_vectors.npy" \
    "$MODEL_DIR/retrieval_metadata.csv" \
    "$MODEL_DIR/retrieval.index" \
    "$MANIFEST"
do
    if [[ ! -s "$required" ]]
    then
        echo "STOP: required artifact is missing or empty: $required"
        exit 1
    fi
done

echo "=== RUN END-TO-END HELD-OUT INFERENCE ==="

apptainer exec --nv "$CONTAINER" "$VENV/bin/python" \
    - "$REPO_DIR" "$MODEL_DIR" "$MANIFEST" "$REPORT" <<'PYSMOKE'
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

repo = Path(sys.argv[1]).resolve()
model_dir = Path(sys.argv[2]).resolve()
manifest_path = Path(sys.argv[3]).resolve()
report_path = Path(sys.argv[4]).resolve()

sys.path.insert(0, str(repo / "src"))

from diptera_id.demo_service import TaxaLensRuntime

frame = pd.read_parquet(manifest_path)

required_columns = {
    "record_id",
    "split",
    "family",
    "genus",
    "species",
    "local_path",
}
missing_columns = required_columns - set(frame.columns)

if missing_columns:
    raise SystemExit(
        "STOP: manifest is missing columns: "
        + ", ".join(sorted(missing_columns))
    )

test = frame[
    frame["split"].fillna("").astype(str).eq("test")
].copy()

test["local_path"] = test["local_path"].fillna("").astype(str)
test = test[
    test["local_path"].map(
        lambda value: bool(value)
        and Path(value).is_file()
        and Path(value).stat().st_size > 0
    )
].copy()

if len(test) < 12:
    raise SystemExit(
        f"STOP: only {len(test):,} usable held-out test images were found"
    )

test["_stable_key"] = test["record_id"].astype(str).map(
    lambda value: hashlib.sha256(
        f"hymenoptera-inference-smoke-v1:{value}".encode("utf-8")
    ).hexdigest()
)

sample = (
    test.sort_values("_stable_key", kind="stable")
    .groupby("family", sort=True, group_keys=False)
    .head(1)
    .sort_values("_stable_key", kind="stable")
    .head(12)
)

if len(sample) != 12:
    raise SystemExit(
        f"STOP: expected 12 family-diverse samples, found {len(sample)}"
    )

print("Loading fused DINOv3 + BioCLIP runtime...", flush=True)

load_started = time.perf_counter()
runtime = TaxaLensRuntime(repo, model_dir).load()
load_seconds = time.perf_counter() - load_started

print(f"Runtime loaded in {load_seconds:.2f} seconds", flush=True)

results = []

for index, row in enumerate(sample.itertuples(index=False), 1):
    image_path = Path(row.local_path)

    with Image.open(image_path) as opened:
        started = time.perf_counter()
        payload = runtime.predict(opened.convert("RGB"), top_k=5)
        inference_seconds = time.perf_counter() - started

    family_predictions = payload.get("predictions", {}).get("family", [])
    top_family = (
        str(family_predictions[0].get("taxon", ""))
        if family_predictions
        else ""
    )
    top_probability = (
        float(family_predictions[0].get("probability", 0.0))
        if family_predictions
        else 0.0
    )

    result = {
        "record_id": str(row.record_id),
        "image_path": str(image_path),
        "true_family": str(row.family),
        "true_genus": str(row.genus),
        "true_species": str(row.species),
        "predicted_family": top_family,
        "family_probability": top_probability,
        "family_correct": top_family == str(row.family),
        "accepted": payload.get("accepted", {}),
        "open_set": payload.get("open_set", {}),
        "neighbours_returned": len(
            payload.get("similar_specimens", [])
        ),
        "inference_seconds": inference_seconds,
    }
    results.append(result)

    marker = "OK" if result["family_correct"] else "MISS"
    print(
        f"{index:02d}/12 {marker}: "
        f"{result['true_family']} -> {top_family} "
        f"({top_probability:.3f}, {inference_seconds:.2f}s)",
        flush=True,
    )

report = {
    "schema": 1,
    "purpose": "end-to-end functional smoke test, not a performance estimate",
    "model": "DINOv3 + BioCLIP fusion",
    "checkpoint": str(model_dir.parent),
    "manifest": str(manifest_path),
    "runtime_load_seconds": load_seconds,
    "samples": len(results),
    "family_top1_correct": sum(
        int(row["family_correct"]) for row in results
    ),
    "family_top1_fraction": (
        sum(int(row["family_correct"]) for row in results) / len(results)
    ),
    "mean_inference_seconds": (
        sum(row["inference_seconds"] for row in results) / len(results)
    ),
    "results": results,
}

pending = report_path.with_suffix(".json.pending")
pending.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
pending.replace(report_path)

print()
print(json.dumps(report, indent=2, ensure_ascii=False))
print(f"\nReport -> {report_path}")
PYSMOKE

echo
echo "=== SMOKE REPORT ==="

python -m json.tool "$REPORT"

echo
echo "Hymenoptera end-to-end inference smoke test completed"
