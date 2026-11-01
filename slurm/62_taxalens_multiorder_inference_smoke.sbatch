#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-multiorder-smoke
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

DATA_ROOT="${DATA_ROOT:-/scratch/$USER/taxalens/v11_hymenoptera_teacher}"
CACHE_ROOT="${TAXALENS_CACHE_ROOT:-/shares/hawlitschek.ieu.uzh/$USER/TaxaLens/shared_backbones/dinov3_bioclip2_v1/cache}"

export HF_HOME="$CACHE_ROOT/huggingface"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export TAXALENS_DIPTERA_MODEL_DIR="/shares/hawlitschek.ieu.uzh/$USER/TaxaLens/v10_raw200k_swiss28/models"
export TAXALENS_HYMENOPTERA_MODEL_DIR="/shares/hawlitschek.ieu.uzh/$USER/TaxaLens/v11_hymenoptera_teacher/hymenoptera_url_baseline_v1/models"
export TAXALENS_ORDER_ROUTER_DIR="/shares/hawlitschek.ieu.uzh/$USER/TaxaLens/multiorder_runtime_v1/order_router_v1"

OUT_DIR="$DATA_ROOT/pipeline/multiorder_inference_smoke_v1"
REPORT="$OUT_DIR/multiorder_inference_smoke.json"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

echo "Compute node: $(hostname)"
echo "Diptera model:     $TAXALENS_DIPTERA_MODEL_DIR"
echo "Hymenoptera model: $TAXALENS_HYMENOPTERA_MODEL_DIR"
echo "Order router:      $TAXALENS_ORDER_ROUTER_DIR"
echo

apptainer exec --nv "$CONTAINER" "$VENV/bin/python" \
    - "$REPO_DIR" "$REPORT" <<'PY'
import inspect
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

from diptera_id.multiorder_service import MultiOrderTaxaLensRuntime


root = Path(sys.argv[1])
report_path = Path(sys.argv[2])

runtime = MultiOrderTaxaLensRuntime.from_environment(root)
runtime.load()

if runtime.diptera._embedder is not runtime.hymenoptera._embedder:
    raise SystemExit(
        "STOP: the two order runtimes do not share one embedder"
    )

image = Image.new("RGB", (640, 480), "white")
draw = ImageDraw.Draw(image)
draw.ellipse((120, 80, 500, 390), fill=(75, 55, 35))
draw.line((180, 130, 60, 30), fill="black", width=8)
draw.line((440, 130, 580, 25), fill="black", width=8)
draw.line((220, 330, 100, 460), fill="black", width=8)
draw.line((400, 330, 540, 455), fill="black", width=8)

parameters = inspect.signature(runtime.predict).parameters
mode_name = (
    "mode"
    if "mode" in parameters
    else "order_mode"
)

results = {}

for mode in ("Auto", "Diptera", "Hymenoptera"):
    started = time.perf_counter()
    payload = runtime.predict(
        image,
        **{mode_name: mode},
    )
    elapsed = time.perf_counter() - started

    order = payload.get("order") or {}
    selected = (
        order.get("selected")
        or order.get("selected_order")
    )
    predictions = payload.get("predictions") or {}
    family = predictions.get("family") or []

    results[mode] = {
        "selected_order": selected,
        "suggested_order": (
            order.get("suggested")
            or order.get("predicted")
            or order.get("router_order")
            or order.get("routed_order")
            or order.get("candidate")
            or order.get("label")
        ),
        "router_confidence": order.get("confidence"),
        "family": (
            family[0].get("taxon")
            if family
            else None
        ),
        "family_probability": (
            family[0].get("probability")
            if family
            else None
        ),
        "inference_seconds": elapsed,
    }

if results["Diptera"]["selected_order"] != "Diptera":
    raise SystemExit("STOP: manual Diptera routing failed")

if results["Hymenoptera"]["selected_order"] != "Hymenoptera":
    raise SystemExit("STOP: manual Hymenoptera routing failed")

report = {
    "schema": 1,
    "shared_embedder": True,
    "router_loaded": True,
    "diptera_loaded": True,
    "hymenoptera_loaded": True,
    "results": results,
}

report_path.parent.mkdir(parents=True, exist_ok=True)
pending = report_path.with_suffix(".json.pending")
pending.write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
pending.replace(report_path)

print(json.dumps(report, indent=2, ensure_ascii=False))
PY

echo
echo "=== RESULT REPORT ==="
python -m json.tool "$REPORT"

echo
echo "Multi-order end-to-end GPU smoke test completed"
