#!/usr/bin/env python3
"""Train conditional family → genus → species heads with centroid open-set gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import Counter

from embedding_safety import cohort_indices, cohort_digest


def deterministic_split(group: str, train_percent: int = 80, val_percent: int = 10) -> str:
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < train_percent:
        return "train"
    return "val" if bucket < train_percent + val_percent else "test"


def source_class_weights(labels: np.ndarray, sources: np.ndarray) -> np.ndarray:
    label_counts = pd.Series(labels).value_counts()
    source_counts = pd.Series(sources).value_counts()
    weights = np.array([
        1.0 / (float(label_counts[label]) * float(source_counts[source])) ** 0.5
        for label, source in zip(labels, sources)
    ])
    weights /= weights.mean()
    return np.clip(weights, 0.1, 10.0)


def centroid_gate(vectors: np.ndarray, labels: np.ndarray, quantile: float, margin: float) -> dict:
    gate = {}
    for label in sorted(set(labels)):
        selected = vectors[labels == label]
        centroid = selected.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        similarities = selected @ centroid
        threshold = float(np.quantile(similarities, quantile) - margin)
        gate[str(label)] = {"centroid": centroid.astype(np.float32), "threshold": max(-1.0, min(1.0, threshold))}
    return gate


def top_k_accuracy(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray, k: int) -> float:
    lookup = {label: index for index, label in enumerate(classes)}
    truth = np.array([lookup[label] for label in y_true])
    winners = np.argsort(-probabilities, axis=1)[:, :k]
    return float(np.mean(np.any(winners == truth[:, None], axis=1)))


def evaluation_breakdown(
    truth: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    sources: np.ndarray,
    source_splits: np.ndarray | None = None,
    dataset_splits: np.ndarray | None = None,
) -> dict:
    by_source: dict[str, dict] = {}
    for source in sorted(set(sources)):
        mask = sources == source
        if not np.any(mask):
            continue
        by_source[str(source)] = {
            "n": int(mask.sum()),
            "top1_accuracy": float(accuracy_score(truth[mask], predicted[mask])),
            "top5_accuracy": top_k_accuracy(truth[mask], probabilities[mask], classes, min(5, len(classes))),
        }
    by_source_split: dict[str, dict] = {}
    if source_splits is not None:
        for split in sorted({str(value) for value in source_splits if str(value)}):
            mask = source_splits == split
            if not np.any(mask):
                continue
            by_source_split[split] = {
                "n": int(mask.sum()),
                "top1_accuracy": float(accuracy_score(truth[mask], predicted[mask])),
                "top5_accuracy": top_k_accuracy(truth[mask], probabilities[mask], classes, min(5, len(classes))),
            }
    errors = Counter((str(t), str(p)) for t, p in zip(truth, predicted) if str(t) != str(p))
    confusion_pairs = [
        {"truth": truth_label, "predicted": predicted_label, "count": int(count)}
        for (truth_label, predicted_label), count in errors.most_common(50)
    ]
    by_dataset_split: dict[str, dict] = {}
    if dataset_splits is not None:
        for split in sorted(set(dataset_splits)):
            split_mask = dataset_splits == split
            if not np.any(split_mask):
                continue
            by_dataset_split[str(split)] = {
                "n": int(split_mask.sum()),
                "top1_accuracy": float(accuracy_score(truth[split_mask], predicted[split_mask])),
                "top5_accuracy": top_k_accuracy(
                    truth[split_mask], probabilities[split_mask], classes, min(5, len(classes))
                ),
            }
    return {
        "by_source": by_source,
        "by_source_split": by_source_split,
        "by_dataset_split": by_dataset_split,
        "top_confusions": confusion_pairs,
    }


def train_head(
    vectors: np.ndarray,
    frame: pd.DataFrame,
    label_column: str,
    mask: pd.Series,
    min_per_class: int,
    gate_quantile: float,
    gate_margin: float,
) -> tuple[LogisticRegression | None, dict, dict]:
    labels = frame[label_column].astype(str)
    train_mask = mask & frame["split"].eq("train") & labels.ne("")
    counts = labels[train_mask].value_counts()
    allowed = set(counts[counts >= min_per_class].index)
    train_indices = np.flatnonzero((train_mask & labels.isin(allowed)).to_numpy())
    if len(train_indices) == 0 or len(set(labels.iloc[train_indices])) < 2:
        return None, {}, {}
    train_labels = labels.iloc[train_indices].to_numpy()
    sources = frame.iloc[train_indices]["source"].astype(str).to_numpy()
    model = LogisticRegression(max_iter=3000, C=2.0)
    model.fit(vectors[train_indices], train_labels, sample_weight=source_class_weights(train_labels, sources))
    gate = centroid_gate(vectors[train_indices], train_labels, gate_quantile, gate_margin)

    eval_mask = mask & frame["split"].isin({"val", "test"}) & labels.isin(model.classes_)
    eval_indices = np.flatnonzero(eval_mask.to_numpy())
    metrics = {"train_specimens": int(len(train_indices)), "evaluation_specimens": int(len(eval_indices)), "classes": int(len(model.classes_))}
    if len(eval_indices):
        truth = labels.iloc[eval_indices].to_numpy()
        probabilities = model.predict_proba(vectors[eval_indices])
        predicted = model.classes_[np.argmax(probabilities, axis=1)]
        metrics.update({
            "top1_accuracy": float(accuracy_score(truth, predicted)),
            "top5_accuracy": top_k_accuracy(truth, probabilities, model.classes_, min(5, len(model.classes_))),
        })
        sources_eval = frame.iloc[eval_indices]["source"].astype(str).to_numpy()
        source_splits = (
            frame.iloc[eval_indices]["source_split"].astype(str).to_numpy()
            if "source_split" in frame.columns
            else None
        )
        dataset_splits = frame.iloc[eval_indices]["split"].astype(str).to_numpy()
        metrics.update(evaluation_breakdown(
            truth, predicted, probabilities, model.classes_, sources_eval, source_splits, dataset_splits
        ))
    return model, gate, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models_microdiptera")
    parser.add_argument("--cohort-manifest", help="Align this model to the exact shared evaluation cohort")
    parser.add_argument("--out", help="Defaults to MODEL_DIR/classifiers.joblib")
    parser.add_argument("--min-family", type=int, default=20)
    parser.add_argument("--min-genus", type=int, default=12)
    parser.add_argument("--min-species", type=int, default=8)
    parser.add_argument("--species-label-quality", default="A,B", help="Add C only after reviewing iNaturalist species labels")
    parser.add_argument("--gate-quantile", type=float, default=0.05)
    parser.add_argument("--gate-margin", type=float, default=0.02)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    frame = pd.read_csv(model_dir / "embedded_manifest.csv", dtype=str, keep_default_na=False)
    vectors = np.load(model_dir / "embeddings.npy").astype(np.float32)
    if len(frame) != len(vectors):
        raise SystemExit("embedded manifest and embeddings.npy have different row counts")
    if args.cohort_manifest:
        cohort = pd.read_csv(args.cohort_manifest, dtype=str, keep_default_na=False)
        indices = cohort_indices(frame, cohort)
        frame = frame.iloc[indices].reset_index(drop=True)
        vectors = vectors[indices]
    vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    if "source" not in frame:
        frame["source"] = "unknown"
    if "split_group" not in frame:
        frame["split_group"] = [f"specimen-{index}" for index in range(len(frame))]
    if "split" not in frame:
        frame["split"] = ""
    missing_split = frame["split"].astype(str).str.strip().eq("")
    frame.loc[missing_split, "split"] = frame.loc[missing_split, "split_group"].astype(str).map(deterministic_split)
    unknown_splits = set(frame["split"].astype(str)) - {"train", "val", "test"}
    if unknown_splits:
        raise SystemExit(f"unknown dataset split values: {sorted(unknown_splits)}")
    if (frame.groupby("split_group")["split"].nunique() > 1).any():
        raise SystemExit("split leakage: the same split_group occurs in multiple dataset splits")

    if "eligible_supervised" in frame:
        all_rows = frame["eligible_supervised"].astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        all_rows = pd.Series([True] * len(frame), index=frame.index)
    family_model, family_gate, family_metrics = train_head(
        vectors, frame, "family", all_rows, args.min_family, args.gate_quantile, args.gate_margin
    )
    if family_model is None:
        raise SystemExit("Could not train family head; collect at least two families with enough train specimens")

    genus_models: dict[str, LogisticRegression] = {}
    genus_gates: dict[str, dict] = {}
    genus_report: dict[str, dict] = {}
    for family in family_model.classes_:
        mask = frame["family"].astype(str).eq(str(family))
        model, gate, metrics = train_head(
            vectors, frame, "genus", mask, args.min_genus, args.gate_quantile, args.gate_margin
        )
        if model is not None:
            genus_models[str(family)] = model
            genus_gates[str(family)] = gate
            genus_report[str(family)] = metrics

    allowed_quality = {value.strip() for value in args.species_label_quality.split(",") if value.strip()}
    species_models: dict[str, LogisticRegression] = {}
    species_gates: dict[str, dict] = {}
    species_report: dict[str, dict] = {}
    quality_mask = frame.get("label_quality", pd.Series([""] * len(frame))).astype(str).isin(allowed_quality)
    for family, genus_model in genus_models.items():
        for genus in genus_model.classes_:
            key = f"{family}\x1f{genus}"
            mask = frame["family"].astype(str).eq(family) & frame["genus"].astype(str).eq(str(genus)) & quality_mask
            model, gate, metrics = train_head(
                vectors, frame, "species", mask, args.min_species, args.gate_quantile, args.gate_margin
            )
            if model is not None:
                species_models[key] = model
                species_gates[key] = gate
                species_report[key] = metrics

    embedding_config = {}
    config_path = model_dir / "embedding_config.json"
    if config_path.exists():
        embedding_config = json.loads(config_path.read_text(encoding="utf-8"))
    evaluation_context = {
        "cohort_sha256": cohort_digest(frame),
        "specimens": int(len(frame)),
        "split_counts": {str(key): int(value) for key, value in frame["split"].value_counts().items()},
        "comparison_rule": "all encoders must use this exact specimen cohort",
    }
    report = {
        "evaluation_context": evaluation_context,
        "family": family_metrics,
        "genus_by_family": genus_report,
        "species_by_genus": species_report,
    }
    payload = {
        "bundle_type": "hierarchical_v1",
        "hierarchical_models": {
            "family": family_model,
            "genus_by_family": genus_models,
            "species_by_genus": species_models,
        },
        "gates": {
            "family": family_gate,
            "genus_by_family": genus_gates,
            "species_by_genus": species_gates,
        },
        "metadata": {
            "training": "hierarchical source-balanced whole-specimen embeddings",
            "embedding": embedding_config,
            "species_label_quality": sorted(allowed_quality),
            "open_set": {"method": "class centroid lower-quantile gate", "quantile": args.gate_quantile, "margin": args.gate_margin},
            "confidence_note": "scores are not taxonomic certainty; fine-rank suggestions require independent verification",
            "evaluation_scope": "family is end-to-end; genus/species head metrics are conditional on the true parent taxon",
            "evaluation": report,
            "evaluation_context": evaluation_context,
        },
    }
    out = Path(args.out) if args.out else model_dir / "classifiers.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out)
    (model_dir / "training_report_hierarchical.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"family classes: {len(family_model.classes_)}")
    print(f"conditional genus heads: {len(genus_models)}")
    print(f"curated species heads: {len(species_models)}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
