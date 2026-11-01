#!/usr/bin/bash -l
#SBATCH --job-name=taxalens-order-router
#SBATCH --account=hawlitschek.ieu.uzh
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load apptainer

REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/projects/TaxaLens}"
CONTAINER="${TAXALENS_CONTAINER:-/scratch/$USER/taxalens/containers/pytorch-2.12.0-cuda13.2.sif}"
VENV="${TAXALENS_VENV:-$HOME/.venvs/taxalens-v09}"

AUDIT="/scratch/$USER/taxalens/v11_hymenoptera_teacher/pipeline/multiorder_runtime_audit_v1/model_bundle_audit.json"

FINAL="/shares/hawlitschek.ieu.uzh/ssagut/TaxaLens/multiorder_runtime_v1/order_router_v1"
STAGING="${FINAL}.pending-${SLURM_JOB_ID}"

export PYTHONPATH="$REPO_DIR/src:$REPO_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$REPO_DIR"

echo "Compute node: $(hostname)"
echo "Audit report: $AUDIT"
echo "Permanent router: $FINAL"
echo

if [[ -d "$FINAL" ]]
then
    (
        cd "$FINAL"
        sha256sum --quiet -c SHA256SUMS.txt
    )

    echo "Existing order-router checkpoint is valid"
    exit 0
fi

test -s "$AUDIT"
mkdir -p "$STAGING"

cleanup() {
    if [[ -d "$STAGING" ]]
    then
        rm -rf -- "$STAGING"
    fi
}

trap cleanup EXIT

apptainer exec "$CONTAINER" "$VENV/bin/python" \
  - "$AUDIT" "$STAGING" <<'PY'
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
)

audit_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])

audit = json.loads(
    audit_path.read_text(encoding="utf-8")
)

def runtime_encoder_config(config):
    backend = str(config.get("backend", ""))

    if backend == "fusion":
        encoders = []

        for encoder in config.get("encoders", []):
            encoders.append({
                "backend": encoder.get("backend"),
                "backbone": encoder.get("backbone"),
                "image_size": int(
                    encoder.get("image_size", 0)
                ),
                "tile_grid": int(
                    encoder.get("tile_grid", 1)
                ),
                "include_whole": bool(
                    encoder.get("include_whole", True)
                ),
                "preprocessing": encoder.get(
                    "preprocessing"
                ),
            })

        return {
            "backend": "fusion",
            "method": config.get("method"),
            "dimension": int(config.get("dimension", 0)),
            "encoders": encoders,
        }

    return {
        "backend": backend,
        "backbone": config.get("backbone"),
        "image_size": int(config.get("image_size", 0)),
        "tile_grid": int(config.get("tile_grid", 1)),
        "include_whole": bool(
            config.get("include_whole", True)
        ),
        "preprocessing": config.get("preprocessing"),
    }

def candidates(order):
    return [
        bundle
        for bundle in audit["bundles"]
        if bundle.get("inferred_order") == order
        and bundle.get("complete_runtime_bundle")
        and bundle.get("loadable")
        and bundle.get("permanent_storage")
        and bundle.get("retrieval_vector_shape")
        and bundle["retrieval_vector_shape"][1] == 1536
        and bundle.get("embedding", {}).get("backend")
            == "fusion"
    ]

diptera_candidates = candidates("Diptera")
hymenoptera_candidates = candidates("Hymenoptera")

if not diptera_candidates:
    raise SystemExit(
        "STOP: no complete permanent Diptera fusion bundle"
    )

if not hymenoptera_candidates:
    raise SystemExit(
        "STOP: no complete permanent Hymenoptera fusion bundle"
    )

diptera = max(
    diptera_candidates,
    key=lambda item: item["retrieval_vector_shape"][0],
)

hymenoptera = max(
    hymenoptera_candidates,
    key=lambda item: item["retrieval_vector_shape"][0],
)

diptera_config = runtime_encoder_config(
    diptera["embedding"]
)
hymenoptera_config = runtime_encoder_config(
    hymenoptera["embedding"]
)

if diptera_config != hymenoptera_config:
    raise SystemExit(
        "STOP: inference-relevant encoder configurations differ\n"
        + json.dumps({
            "Diptera": diptera_config,
            "Hymenoptera": hymenoptera_config,
        }, indent=2)
    )

print("Shared encoder compatibility verified")
print(f"Diptera bundle:     {diptera['model_dir']}")
print(f"Hymenoptera bundle: {hymenoptera['model_dir']}")
print(json.dumps(diptera_config, indent=2))

rng = np.random.default_rng(42)

def choose(indices, limit):
    indices = np.asarray(indices, dtype=np.int64)

    if len(indices) <= limit:
        return indices

    return np.sort(
        rng.choice(indices, size=limit, replace=False)
    )

def load_order(order, bundle):
    directory = Path(bundle["model_dir"])

    vectors = np.load(
        directory / "retrieval_vectors.npy",
        mmap_mode="r",
    )

    metadata = pd.read_csv(
        directory / "retrieval_metadata.csv",
        low_memory=False,
    )

    if len(metadata) != len(vectors):
        raise RuntimeError(
            f"{order}: metadata/vector row mismatch"
        )

    split_column = next(
        (
            name for name in ("split", "source_split")
            if name in metadata.columns
        ),
        None,
    )

    if split_column is None:
        if "record_id" not in metadata.columns:
            preferred_columns = (
                "specimen_id",
                "specimen_key",
                "source_record_id",
                "image_id",
                "local_path",
                "image_path",
                "path",
                "filename",
            )

            ordered_columns = list(preferred_columns) + [
                column
                for column in metadata.columns
                if column not in preferred_columns
            ]

            router_key_column = None

            for column in ordered_columns:
                if column not in metadata.columns:
                    continue

                values = (
                    metadata[column]
                    .fillna("")
                    .astype(str)
                )

                if (
                    values.ne("").all()
                    and values.nunique() == len(metadata)
                ):
                    router_key_column = column
                    break

            if router_key_column is None:
                router_keys = metadata.index.to_series().map(
                    lambda value: f"{order}:row:{value}"
                )
                router_key_source = "stable vector-row index"
            else:
                router_keys = (
                    metadata[router_key_column]
                    .fillna("")
                    .astype(str)
                    .map(lambda value: f"{order}:{value}")
                )
                router_key_source = router_key_column

            metadata["record_id"] = router_keys.to_numpy()

            print(
                f"{order}: generated router IDs from "
                f"{router_key_source}"
            )

        def deterministic_router_split(value: object) -> str:
            digest = hashlib.sha256(
                str(value).encode("utf-8")
            ).digest()
            bucket = int.from_bytes(digest[:8], "big") % 10

            if bucket < 8:
                return "train"
            if bucket == 8:
                return "val"
            return "test"

        metadata["router_split"] = (
            metadata["record_id"]
            .fillna("")
            .astype(str)
            .map(deterministic_router_split)
        )
        split_column = "router_split"

        print(
            f"{order}: original split unavailable; "
            "using deterministic 80/10/10 record-ID split"
        )
    splits = (
        metadata[split_column]
        .fillna("")
        .astype(str)
        .str.lower()
        .to_numpy()
    )

    train_idx = choose(
        np.flatnonzero(splits == "train"),
        50_000,
    )
    val_idx = choose(
        np.flatnonzero(splits == "val"),
        10_000,
    )
    test_idx = choose(
        np.flatnonzero(splits == "test"),
        20_000,
    )

    for split_name, indices in (
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ):
        if len(indices) == 0:
            raise RuntimeError(
                f"{order}: empty {split_name} split"
            )

    def materialize(indices):
        array = np.asarray(
            vectors[indices],
            dtype=np.float32,
        )
        norms = np.linalg.norm(
            array,
            axis=1,
            keepdims=True,
        )
        return array / np.clip(norms, 1e-12, None)

    return {
        "order": order,
        "model_dir": str(directory),
        "total_rows": int(len(vectors)),
        "train": materialize(train_idx),
        "val": materialize(val_idx),
        "test": materialize(test_idx),
    }

with ThreadPoolExecutor(max_workers=2) as pool:
    diptera_future = pool.submit(
        load_order,
        "Diptera",
        diptera,
    )
    hymenoptera_future = pool.submit(
        load_order,
        "Hymenoptera",
        hymenoptera,
    )

    datasets = [
        diptera_future.result(),
        hymenoptera_future.result(),
    ]

def combine(split):
    arrays = [
        dataset[split]
        for dataset in datasets
    ]

    labels = [
        np.full(
            len(dataset[split]),
            dataset["order"],
            dtype=object,
        )
        for dataset in datasets
    ]

    return (
        np.concatenate(arrays, axis=0),
        np.concatenate(labels, axis=0),
    )

X_train, y_train = combine("train")
X_val, y_val = combine("val")
X_test, y_test = combine("test")

print(
    f"Router cohorts: train={len(y_train):,}, "
    f"val={len(y_val):,}, test={len(y_test):,}"
)

model = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=1e-5,
    max_iter=100,
    tol=1e-5,
    class_weight="balanced",
    random_state=42,
    average=True,
    n_jobs=-1,
)

model.fit(X_train, y_train)

val_probabilities = model.predict_proba(X_val)
val_predictions = model.classes_[
    np.argmax(val_probabilities, axis=1)
]
val_confidence = np.max(
    val_probabilities,
    axis=1,
)
val_correct = val_predictions == y_val

threshold = 0.95
threshold_search = []

for candidate in np.linspace(0.50, 0.99, 100):
    accepted = val_confidence >= candidate

    if not accepted.any():
        continue

    precision = float(
        val_correct[accepted].mean()
    )
    coverage = float(accepted.mean())

    threshold_search.append({
        "threshold": float(candidate),
        "accepted_accuracy": precision,
        "coverage": coverage,
    })

eligible = [
    item for item in threshold_search
    if item["accepted_accuracy"] >= 0.995
]

if eligible:
    threshold = min(
        eligible,
        key=lambda item: item["threshold"],
    )["threshold"]

test_probabilities = model.predict_proba(X_test)
test_predictions = model.classes_[
    np.argmax(test_probabilities, axis=1)
]
test_confidence = np.max(
    test_probabilities,
    axis=1,
)
test_accepted = test_confidence >= threshold

test_accuracy = float(
    accuracy_score(y_test, test_predictions)
)

accepted_accuracy = (
    float(
        accuracy_score(
            y_test[test_accepted],
            test_predictions[test_accepted],
        )
    )
    if test_accepted.any()
    else None
)

accepted_coverage = float(
    test_accepted.mean()
)

encoder_signature = hashlib.sha256(
    json.dumps(
        diptera_config,
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()

payload = {
    "bundle_type": "taxalens_order_router_v1",
    "model": model,
    "classes": [
        str(value)
        for value in model.classes_
    ],
    "acceptance_threshold": float(threshold),
    "encoder_config": diptera_config,
    "encoder_signature_sha256": encoder_signature,
}

joblib.dump(
    payload,
    out_dir / "order_router.joblib",
    compress=3,
)

report = {
    "schema": 1,
    "profile": "taxalens-diptera-hymenoptera-router-v1",
    "diptera_model_dir": diptera["model_dir"],
    "hymenoptera_model_dir": hymenoptera["model_dir"],
    "shared_encoder_config": diptera_config,
    "encoder_signature_sha256": encoder_signature,
    "training_rows": int(len(y_train)),
    "validation_rows": int(len(y_val)),
    "test_rows": int(len(y_test)),
    "sampled_rows_by_order": {
        dataset["order"]: {
            "available": dataset["total_rows"],
            "train": len(dataset["train"]),
            "val": len(dataset["val"]),
            "test": len(dataset["test"]),
        }
        for dataset in datasets
    },
    "classes": [
        str(value)
        for value in model.classes_
    ],
    "test_accuracy": test_accuracy,
    "acceptance_threshold": float(threshold),
    "accepted_test_accuracy": accepted_accuracy,
    "accepted_test_coverage": accepted_coverage,
    "confusion_matrix_labels": [
        str(value)
        for value in model.classes_
    ],
    "confusion_matrix": confusion_matrix(
        y_test,
        test_predictions,
        labels=model.classes_,
    ).tolist(),
    "test_log_loss": float(
        log_loss(
            y_test,
            test_probabilities,
            labels=model.classes_,
        )
    ),
}

(out_dir / "order_router_report.json").write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    ) + "\n",
    encoding="utf-8",
)

print(json.dumps(report, indent=2, ensure_ascii=False))
PY

echo
echo "=== CREATE AND VERIFY CHECKSUMS ==="

(
    cd "$STAGING"

    sha256sum \
      order_router.joblib \
      order_router_report.json \
      > SHA256SUMS.txt

    sha256sum --quiet -c SHA256SUMS.txt
)

mkdir -p "$(dirname "$FINAL")"
mv "$STAGING" "$FINAL"

trap - EXIT

(
    cd "$FINAL"
    sha256sum --quiet -c SHA256SUMS.txt
)

echo
echo "=== PERMANENT ORDER ROUTER ==="

ls -lh "$FINAL"
cat "$FINAL/order_router_report.json"

echo
echo "TaxaLens Diptera/Hymenoptera order router completed"
