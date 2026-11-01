#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-multiorder-audit
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

OUT_DIR="/scratch/$USER/taxalens/v11_hymenoptera_teacher/pipeline/multiorder_runtime_audit_v1"
REPORT="$OUT_DIR/model_bundle_audit.json"

mkdir -p "$OUT_DIR"

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "No models or training files will be modified"
echo

apptainer exec "$CONTAINER" "$VENV/bin/python" \
  - "$REPORT" \
  "/shares/hawlitschek.ieu.uzh/ssagut/TaxaLens" \
  "/scratch/$USER/taxalens" \
  "$REPO_DIR/models" <<'PY'
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from diptera_id.hierarchy import load_classifier_bundle

report_path = Path(sys.argv[1])
roots = [Path(value) for value in sys.argv[2:]]

skip_directories = {
    ".git",
    "cache",
    "containers",
    "image_cache",
    "images",
    "selected_images",
    "tmp",
    "__pycache__",
}

def scan_root(root):
    found = []

    if not root.is_dir():
        return found

    for directory, names, files in os.walk(root):
        names[:] = [
            name for name in names
            if name not in skip_directories
        ]

        if "classifiers.joblib" in files:
            found.append(
                Path(directory) / "classifiers.joblib"
            )

    return found


active_roots = [
    root for root in roots
    if root.is_dir()
]

workers = min(4, max(1, len(active_roots)))
classifier_paths = []

with ThreadPoolExecutor(max_workers=workers) as pool:
    for discovered in pool.map(scan_root, active_roots):
        classifier_paths.extend(discovered)

classifier_paths = sorted({
    path.resolve()
    for path in classifier_paths
})

print(
    f"Scanned {len(active_roots)} roots "
    f"with {workers} parallel workers"
)

diptera_markers = {
    "Culicidae",
    "Muscidae",
    "Phoridae",
    "Syrphidae",
    "Tachinidae",
}

hymenoptera_markers = {
    "Apidae",
    "Braconidae",
    "Formicidae",
    "Ichneumonidae",
    "Vespidae",
}

required = (
    "classifiers.joblib",
    "retrieval_vectors.npy",
    "retrieval_metadata.csv",
)

def clean(value):
    if isinstance(value, dict):
        return {
            str(key): clean(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

bundles = []

for classifier_path in classifier_paths:
    directory = classifier_path.parent
    missing = [
        name for name in required
        if not (directory / name).is_file()
    ]

    item = {
        "model_dir": str(directory),
        "permanent_storage": str(directory).startswith("/shares/"),
        "missing_required_files": missing,
        "complete_runtime_bundle": not missing,
    }

    try:
        bundle = load_classifier_bundle(classifier_path)
        families = sorted(
            str(value)
            for value in bundle.family_model.classes_
        )

        diptera_score = len(set(families) & diptera_markers)
        hymenoptera_score = len(
            set(families) & hymenoptera_markers
        )

        if diptera_score > hymenoptera_score:
            inferred_order = "Diptera"
        elif hymenoptera_score > diptera_score:
            inferred_order = "Hymenoptera"
        else:
            inferred_order = "unknown"

        embedding = clean(
            bundle.metadata.get("embedding", {})
        )

        vectors_path = directory / "retrieval_vectors.npy"
        vector_shape = None
        vector_dtype = None

        if vectors_path.is_file():
            vectors = np.load(vectors_path, mmap_mode="r")
            vector_shape = list(vectors.shape)
            vector_dtype = str(vectors.dtype)

        signature_payload = {
            "embedding": embedding,
            "vector_dimension": (
                vector_shape[1]
                if vector_shape and len(vector_shape) == 2
                else None
            ),
        }

        signature = hashlib.sha256(
            json.dumps(
                signature_payload,
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

        item.update({
            "loadable": True,
            "inferred_order": inferred_order,
            "family_classes": len(families),
            "family_examples": families[:12],
            "genus_heads": len(
                getattr(bundle, "genus_by_family", {})
            ),
            "species_heads": len(
                getattr(bundle, "species_by_genus", {})
            ),
            "embedding": embedding,
            "retrieval_vector_shape": vector_shape,
            "retrieval_vector_dtype": vector_dtype,
            "embedding_signature_sha256": signature,
            "metadata_keys": sorted(
                str(key)
                for key in bundle.metadata
            ),
        })

    except Exception as exc:
        item.update({
            "loadable": False,
            "load_error": (
                f"{type(exc).__name__}: {exc}"
            ),
        })

    bundles.append(item)

signature_counts = Counter(
    item["embedding_signature_sha256"]
    for item in bundles
    if item.get("loadable")
)

report = {
    "schema": 1,
    "searched_roots": [
        str(root)
        for root in roots
        if root.is_dir()
    ],
    "classifier_files_found": len(classifier_paths),
    "complete_runtime_bundles": sum(
        bool(item["complete_runtime_bundle"])
        for item in bundles
    ),
    "loadable_bundles": sum(
        bool(item.get("loadable"))
        for item in bundles
    ),
    "orders_found": dict(Counter(
        item.get("inferred_order", "unloadable")
        for item in bundles
    )),
    "embedding_signature_counts": dict(signature_counts),
    "bundles": bundles,
    "models_modified": False,
}

temporary = report_path.with_suffix(".json.pending")
temporary.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)
temporary.replace(report_path)

print(json.dumps(report, indent=2, ensure_ascii=False))
PY

echo
echo "=== AUDIT REPORT ==="

ls -lh "$REPORT"

echo
echo "TaxaLens multi-order model audit completed"
echo "No model files were modified"
