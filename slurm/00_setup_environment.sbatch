#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-setup
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
RUNTIME_DIR="/scratch/$USER/taxalens"
CONTAINER_DIR="$RUNTIME_DIR/containers"
APPTAINER_CACHE="$RUNTIME_DIR/apptainer-cache"
APPTAINER_TMP="$RUNTIME_DIR/apptainer-tmp"
PIP_CACHE="$RUNTIME_DIR/pip-cache"
UV_CACHE="$RUNTIME_DIR/uv-cache"

CONTAINER="$CONTAINER_DIR/pytorch-2.12.0-cuda13.2.sif"
VENV="$HOME/.venvs/taxalens-v09"
UV_BIN="$HOME/.local/bin/uv"

mkdir -p \
    "$CONTAINER_DIR" \
    "$APPTAINER_CACHE" \
    "$APPTAINER_TMP" \
    "$PIP_CACHE" \
    "$UV_CACHE" \
    "$(dirname "$VENV")"

export APPTAINER_CACHEDIR="$APPTAINER_CACHE"
export APPTAINER_TMPDIR="$APPTAINER_TMP"
export PIP_CACHE_DIR="$PIP_CACHE"
export UV_CACHE_DIR="$UV_CACHE"
export UV_NO_MODIFY_PATH=1

cd "$REPO_DIR"

if [[ ! -s "$CONTAINER" ]]; then
    apptainer pull "$CONTAINER" \
        docker://pytorch/pytorch:2.12.0-cuda13.2-cudnn9-runtime
fi

if [[ ! -x "$UV_BIN" ]]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

"$UV_BIN" --version

if ! apptainer exec "$CONTAINER" \
    "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then

    if [[ -d "$VENV" ]]; then
        mv "$VENV" "${VENV}.incomplete-${SLURM_JOB_ID:-manual}"
    fi

    apptainer exec "$CONTAINER" \
        "$UV_BIN" venv \
        --python python3 \
        --system-site-packages \
        --seed \
        "$VENV"
fi

apptainer exec "$CONTAINER" \
    "$VENV/bin/python" -m pip install --upgrade pip

apptainer exec "$CONTAINER" \
    "$VENV/bin/python" -m pip install \
    -r "$REPO_DIR/requirements-cluster.txt"

apptainer exec "$CONTAINER" "$VENV/bin/python" - <<'PY'
import sys

import faiss
import numpy
import open_clip
import pandas
import pyarrow
import sklearn
import torch
import transformers

print("TaxaLens environment is ready")
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA will be tested in the next GPU job")
PY
