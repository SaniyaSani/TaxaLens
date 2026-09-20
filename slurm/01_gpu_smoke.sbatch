#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-gpu-smoke
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=lowprio
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gpus=L4:1
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif"
VENV="$HOME/.venvs/taxalens-v09"

cd "$REPO_DIR"

if [[ ! -s "$CONTAINER" ]]; then
    echo "Container not found: $CONTAINER"
    exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "Python environment not found: $VENV"
    exit 1
fi

echo "Compute node: $(hostname)"
nvidia-smi

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

apptainer exec --nv "$CONTAINER" "$VENV/bin/python" - <<'PY'
import torch
import diptera_id

print("TaxaLens package:", diptera_id.__file__)
print("PyTorch:", torch.__version__)
print("Container CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the container")

device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))

a = torch.randn((2048, 2048), device=device)
b = torch.randn((2048, 2048), device=device)
c = a @ b
torch.cuda.synchronize()

print("GPU calculation successful")
print("Result norm:", c.norm().item())
PY
