#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-multiorder-app
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"
PIP_CACHE="${PIP_CACHE_DIR:-/scratch/$USER/taxalens/cache/pip}"

mkdir -p "$PIP_CACHE"
export PIP_CACHE_DIR="$PIP_CACHE"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Repository:   $REPO_DIR"
echo

echo "=== INSTALL DECLARED UI DEPENDENCY ==="

apptainer exec "$CONTAINER" \
    "$VENV/bin/python" -m pip install \
    --upgrade-strategy only-if-needed \
    'gradio>=5.49,<7'

echo
echo "=== VERIFY GRADIO ==="

apptainer exec "$CONTAINER" \
    "$VENV/bin/python" - <<'PY'
import gradio

print("Gradio version:", gradio.__version__)
PY

export TAXALENS_DIPTERA_MODEL_DIR="/shares/hawlitschek.ieu.uzh/$USER/TaxaLens/v10_raw200k_swiss28/models"
export TAXALENS_HYMENOPTERA_MODEL_DIR="/shares/hawlitschek.ieu.uzh/$USER/TaxaLens/v11_hymenoptera_teacher/hymenoptera_url_baseline_v1/models"
export TAXALENS_ORDER_ROUTER_DIR="/shares/hawlitschek.ieu.uzh/$USER/TaxaLens/multiorder_runtime_v1/order_router_v1"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

echo
echo "=== IMPORT COMBINED SITE ==="

apptainer exec "$CONTAINER" \
    "$VENV/bin/python" - <<'PY'
import app

assert app.RUNTIME is not None
assert app.STARTUP_ERROR is None
assert app.demo is not None

print(
    "Combined Diptera + Hymenoptera "
    "Gradio site imports successfully"
)
PY

echo
echo "=== RUN FULL TEST SUITE ==="

apptainer exec "$CONTAINER" \
    "$VENV/bin/python" -m pytest -q

echo
echo "=== FINAL SOURCE CHECKS ==="

apptainer exec "$CONTAINER" \
    "$VENV/bin/python" -m py_compile \
    app.py \
    src/diptera_id/demo_service.py \
    src/diptera_id/multiorder_service.py \
    tests/test_multiorder_runtime.py

bash -n \
    slurm/60_taxalens_multiorder_bundle_audit.sbatch \
    slurm/61_taxalens_order_router.sbatch \
    slurm/62_taxalens_multiorder_inference_smoke.sbatch \
    slurm/63_taxalens_multiorder_app_validation.sbatch

git diff --check

echo
echo "=== STAGE VERIFIED PIPELINE ==="

git add \
    app.py \
    src/diptera_id/demo_service.py \
    src/diptera_id/multiorder_service.py \
    tests/test_multiorder_runtime.py \
    slurm/60_taxalens_multiorder_bundle_audit.sbatch \
    slurm/61_taxalens_order_router.sbatch \
    slurm/62_taxalens_multiorder_inference_smoke.sbatch \
    slurm/63_taxalens_multiorder_app_validation.sbatch

git diff --cached --stat

echo
echo "=== COMMIT ==="

if git diff --cached --quiet
then
    echo "Nothing new to commit"
else
    git commit \
      -m "Add unified Diptera and Hymenoptera inference runtime"
fi

echo
echo "=== FINAL GIT STATE ==="

git status --short --branch
git log -5 --oneline --decorate

echo
echo "Multi-order application validation completed"
echo "The verified commit is ready to push"
