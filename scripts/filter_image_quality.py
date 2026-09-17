#!/usr/bin/env python3
"""Build a conservative, reviewable image-quality gate before embedding.

The original cache is never modified.  Standardized BIOSCAN and field
iNaturalist images receive technical checks; heterogeneous GBIF/DiSSCo museum
media additionally require an OWLv2 fly detection.  Small but sufficiently
detailed detections are cropped into a separate derived cache.  Ambiguous rows
are quarantined instead of silently entering supervised training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diptera_id.corpus.io import load_manifest
from diptera_id.corpus.schema import MASTER_COLUMNS, as_bool, finalize_record


DEFAULT_PROMPTS = (
    "a fly",
    "a pinned fly",
    "a mosquito",
    "a pinned mosquito",
    "a gnat",
    "a midge",
)


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class TechnicalMetrics:
    width: int
    height: int
    contrast_std: float
    entropy: float


@dataclass(frozen=True)
class QualityDecision:
    decision: str
    reason: str
    detection_label: str = ""
    detection_score: float = 0.0
    specimen_area_ratio: float = 0.0
    specimen_short_side_px: float = 0.0
    box_x1: float = math.nan
    box_y1: float = math.nan
    box_x2: float = math.nan
    box_y2: float = math.nan


class BatchDetector(Protocol):
    def detect(self, images: Sequence[Image.Image]) -> list[Detection | None]: ...


def atomic_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name + f".{uuid.uuid4().hex}.pending")
    pending.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(destination)


def atomic_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name + f".{uuid.uuid4().hex}.pending")
    frame.to_parquet(pending, index=False)
    pending.replace(destination)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_source(value: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-") or "unknown"


def technical_metrics(image: Image.Image) -> TechnicalMetrics:
    gray = np.asarray(image.convert("L").resize((128, 128)), dtype=np.uint8)
    histogram = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / histogram.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    return TechnicalMetrics(
        width=int(image.width),
        height=int(image.height),
        contrast_std=float(gray.std()),
        entropy=entropy,
    )


def load_processing_image(path: Path, max_side: int = 4096) -> Image.Image:
    """Decode a bounded proxy; huge originals remain untouched on disk."""
    with Image.open(path) as opened:
        if max(opened.size) > max_side:
            # JPEG decoders can reduce during decoding, avoiding a very large RGB allocation.
            opened.draft("RGB", (max_side, max_side))
        image = opened.convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def clipped_box(detection: Detection, metrics: TechnicalMetrics) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = detection.box
    x1 = max(0.0, min(float(metrics.width), x1))
    x2 = max(0.0, min(float(metrics.width), x2))
    y1 = max(0.0, min(float(metrics.height), y1))
    y2 = max(0.0, min(float(metrics.height), y2))
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def decide_quality(
    source: str,
    metrics: TechnicalMetrics,
    detection: Detection | None,
    *,
    strict_sources: set[str],
    min_image_side: int,
    min_contrast_std: float,
    min_entropy: float,
    min_specimen_side: int,
    min_specimen_area_ratio: float,
    crop_below_area_ratio: float,
) -> QualityDecision:
    if min(metrics.width, metrics.height) < min_image_side:
        return QualityDecision("exclude", "image_too_small")
    if metrics.contrast_std < min_contrast_std or metrics.entropy < min_entropy:
        return QualityDecision("exclude", "blank_or_near_uniform")
    if source not in strict_sources:
        return QualityDecision("accept", "trusted_source_technical_pass")
    if detection is None:
        return QualityDecision("review", "no_fly_detected")

    x1, y1, x2, y2 = clipped_box(detection, metrics)
    box_width, box_height = x2 - x1, y2 - y1
    short_side = min(box_width, box_height)
    area_ratio = box_width * box_height / max(1.0, metrics.width * metrics.height)
    common = {
        "detection_label": detection.label,
        "detection_score": float(detection.score),
        "specimen_area_ratio": float(area_ratio),
        "specimen_short_side_px": float(short_side),
        "box_x1": x1,
        "box_y1": y1,
        "box_x2": x2,
        "box_y2": y2,
    }
    if short_side < min_specimen_side or area_ratio < min_specimen_area_ratio:
        return QualityDecision("review", "specimen_too_small", **common)
    if area_ratio < crop_below_area_ratio:
        return QualityDecision("accept_crop", "detected_fly_cropped", **common)
    return QualityDecision("accept", "detected_fly_large_enough", **common)


class Owlv2FlyDetector:
    """Text-conditioned fly locator, loaded only by the quality stage."""

    def __init__(
        self,
        model_name: str,
        prompts: Sequence[str],
        score_threshold: float,
        allow_cpu: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
        except ImportError as exc:
            raise SystemExit(
                "Quality stage requires torch and transformers>=4.56. "
                "Install them, then rerun; the cached JPEG files are preserved."
            ) from exc
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif allow_cpu:
            device = torch.device("cpu")
        else:
            raise SystemExit(
                "STOP: OWLv2 quality filtering needs a GPU for this corpus. "
                "In Colab choose Runtime > Change runtime type > T4 GPU, rerun setup, then rerun quality."
            )
        self.torch = torch
        self.device = device
        self.prompts = list(prompts)
        self.score_threshold = float(score_threshold)
        print(f"Loading image-quality detector {model_name} on {device}…", flush=True)
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device).eval()

    def detect(self, images: Sequence[Image.Image]) -> list[Detection | None]:
        if not images:
            return []
        text_labels = [self.prompts for _ in images]
        inputs = self.processor(text=text_labels, images=list(images), return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = self.torch.tensor(
            [(image.height, image.width) for image in images], device=self.device
        )
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.score_threshold,
            text_labels=text_labels,
        )
        chosen: list[Detection | None] = []
        for result in results:
            if len(result["scores"]) == 0:
                chosen.append(None)
                continue
            best = int(result["scores"].argmax().item())
            chosen.append(
                Detection(
                    label=str(result["text_labels"][best]),
                    score=float(result["scores"][best].item()),
                    box=tuple(float(value) for value in result["boxes"][best].tolist()),
                )
            )
        return chosen


def crop_detection(
    image: Image.Image,
    decision: QualityDecision,
    destination: Path,
    padding: float,
) -> tuple[str, str]:
    x1, y1, x2, y2 = decision.box_x1, decision.box_y1, decision.box_x2, decision.box_y2
    width, height = x2 - x1, y2 - y1
    x1 = max(0, int(math.floor(x1 - width * padding)))
    y1 = max(0, int(math.floor(y1 - height * padding)))
    x2 = min(image.width, int(math.ceil(x2 + width * padding)))
    y2 = min(image.height, int(math.ceil(y2 + height * padding)))
    cropped = image.crop((x1, y1, x2, y2))
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name + f".{uuid.uuid4().hex}.pending")
    cropped.save(pending, "JPEG", quality=95)
    pending.replace(destination)
    return str(destination), file_sha256(destination)


def checkpoint_key(frame: pd.DataFrame) -> str:
    values = frame["record_id"].astype(str).tolist()
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def load_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("record_id,decision\n", encoding="utf-8")
        return {}
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"record_id", "decision"}
    if not required.issubset(table.columns):
        raise SystemExit(f"manual override file must contain {sorted(required)}: {path}")
    allowed = {"accept", "exclude"}
    invalid = sorted(set(table["decision"]) - allowed - {""})
    if invalid:
        raise SystemExit(f"manual override decisions must be accept/exclude, got: {invalid}")
    return dict(zip(table["record_id"], table["decision"]))


def manifest_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=MASTER_COLUMNS)
    normalized = pd.DataFrame([finalize_record(row) for row in frame.to_dict(orient="records")])
    return normalized[MASTER_COLUMNS].astype(str)


def write_contact_sheet(
    frame: pd.DataFrame,
    decisions: pd.DataFrame,
    destination: Path,
    limit: int = 40,
) -> int:
    candidates = decisions[decisions["decision"].isin(["review", "exclude"])].head(limit)
    tiles: list[Image.Image] = []
    font = ImageFont.load_default()
    for item in candidates.to_dict(orient="records"):
        source_row = frame.iloc[int(item["input_index"])]
        path = Path(str(source_row.get("local_path", "")))
        if not path.is_file():
            continue
        try:
            image = load_processing_image(path)
            image.thumbnail((256, 166))
        except Exception:
            continue
        tile = Image.new("RGB", (272, 210), "white")
        x, y = (272 - image.width) // 2, 8 + (166 - image.height) // 2
        tile.paste(image, (x, y))
        if not pd.isna(item.get("box_x1")):
            original_width = max(1.0, float(item.get("width", 1)))
            original_height = max(1.0, float(item.get("height", 1)))
            draw = ImageDraw.Draw(tile)
            draw.rectangle(
                (
                    x + float(item["box_x1"]) * image.width / original_width,
                    y + float(item["box_y1"]) * image.height / original_height,
                    x + float(item["box_x2"]) * image.width / original_width,
                    y + float(item["box_y2"]) * image.height / original_height,
                ),
                outline="#ff2f6d",
                width=3,
            )
        draw = ImageDraw.Draw(tile)
        caption = f"{source_row.get('source', '')}: {item['reason']}"
        draw.text((8, 180), caption[:46], fill="black", font=font)
        tiles.append(tile)
    if not tiles:
        return 0
    columns = 4
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * 272, rows * 210), "#dddddd")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 272, (index // columns) * 210))
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name + f".{uuid.uuid4().hex}.pending")
    sheet.save(pending, "JPEG", quality=92)
    pending.replace(destination)
    return len(tiles)


def run_filter(
    *,
    manifest: Path,
    out: Path,
    review_out: Path,
    decisions_out: Path,
    report_out: Path,
    preview_out: Path,
    crop_root: Path,
    checkpoint_dir: Path,
    overrides_path: Path,
    settings: dict,
    detector: BatchDetector | None = None,
    allow_cpu: bool = False,
    required_sources: set[str] | None = None,
) -> dict:
    frame = load_manifest(manifest).fillna("").reset_index(drop=True)
    if frame.empty:
        raise SystemExit("cached training plan is empty")
    if frame["record_id"].astype(str).duplicated().any():
        raise SystemExit("cached training plan contains duplicate record_id values")

    strict_sources = set(settings["strict_sources"])
    prompts = tuple(settings.get("prompts", DEFAULT_PROMPTS))
    settings_record = {
        "gate_version": 1,
        **settings,
        "prompts": list(prompts),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": file_sha256(manifest),
        "rows": int(len(frame)),
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved_settings = checkpoint_dir / "settings.json"
    if saved_settings.exists():
        previous = json.loads(saved_settings.read_text(encoding="utf-8"))
        if previous != settings_record:
            raise SystemExit(
                f"STOP: quality settings/input changed; existing checkpoints were preserved at {checkpoint_dir}. "
                "Use a new checkpoint directory or restore the previous settings."
            )
    else:
        atomic_json(saved_settings, settings_record)

    # Loading OWLv2 is expensive and requires a GPU.  Delay it until an
    # unfinished strict-source batch actually needs inference.  A completed
    # checkpoint run can therefore be finalized later on CPU without loading
    # model weights again.
    active_detector = detector

    checkpoint_size = int(settings["checkpoint_size"])
    batch_size = int(settings["batch_size"])
    checkpoint_paths: list[Path] = []
    total_chunks = math.ceil(len(frame) / checkpoint_size)
    for chunk_index, start in enumerate(range(0, len(frame), checkpoint_size)):
        chunk = frame.iloc[start : start + checkpoint_size]
        checkpoint = checkpoint_dir / f"chunk_{chunk_index:05d}.parquet"
        checkpoint_paths.append(checkpoint)
        expected_key = checkpoint_key(chunk)
        if checkpoint.exists():
            saved = pd.read_parquet(checkpoint)
            if len(saved) != len(chunk) or set(saved["checkpoint_key"]) != {expected_key}:
                raise SystemExit(f"STOP: incompatible quality checkpoint preserved: {checkpoint}")
            print(f"quality checkpoint {chunk_index + 1}/{total_chunks}: reused", flush=True)
            continue

        rows: dict[int, dict] = {}
        strict_items: list[tuple[int, str, TechnicalMetrics, Image.Image]] = []

        def flush_strict_items() -> None:
            """Run one tiny detector batch and immediately release decoded pixels."""
            nonlocal active_detector
            if not strict_items:
                return
            items = list(strict_items)
            images = [item[3] for item in items]
            try:
                if active_detector is None:
                    active_detector = Owlv2FlyDetector(
                        settings["model"], prompts, float(settings["score_threshold"]),
                        allow_cpu=allow_cpu,
                    )
                detections = active_detector.detect(images)
                if len(detections) != len(items):
                    raise RuntimeError("quality detector returned the wrong number of results")
                for (input_index, source, metrics, image), detection in zip(items, detections):
                    row = frame.iloc[input_index]
                    decision = decide_quality(
                        source, metrics, detection, strict_sources=strict_sources,
                        min_image_side=int(settings["min_image_side"]),
                        min_contrast_std=float(settings["min_contrast_std"]),
                        min_entropy=float(settings["min_entropy"]),
                        min_specimen_side=int(settings["min_specimen_side"]),
                        min_specimen_area_ratio=float(settings["min_specimen_area_ratio"]),
                        crop_below_area_ratio=float(settings["crop_below_area_ratio"]),
                    )
                    base = {
                        "input_index": int(input_index),
                        "record_id": str(row.get("record_id", "")),
                        "source": source,
                        "checkpoint_key": expected_key,
                        "updated_local_path": "",
                        "updated_image_sha256": "",
                    }
                    if decision.decision == "accept_crop":
                        destination = (
                            crop_root / safe_source(source) / str(row["record_id"])[:2]
                            / f"{row['record_id']}.jpg"
                        )
                        local_path, sha256 = crop_detection(
                            image, decision, destination, float(settings["crop_padding"])
                        )
                        base["updated_local_path"] = local_path
                        base["updated_image_sha256"] = sha256
                    rows[input_index] = {**base, **asdict(decision), **asdict(metrics)}
            finally:
                for image in images:
                    image.close()
                strict_items.clear()
        for input_index, row in chunk.iterrows():
            source = str(row.get("source", ""))
            local_path = Path(str(row.get("local_path", "")))
            base = {
                "input_index": int(input_index),
                "record_id": str(row.get("record_id", "")),
                "source": source,
                "checkpoint_key": expected_key,
                "updated_local_path": "",
                "updated_image_sha256": "",
            }
            if not as_bool(row.get("eligible_supervised", True)):
                rows[input_index] = {
                    **base,
                    **asdict(QualityDecision("exclude", "upstream_ineligible")),
                    "width": 0,
                    "height": 0,
                    "contrast_std": 0.0,
                    "entropy": 0.0,
                }
                continue
            try:
                image = load_processing_image(local_path, int(settings.get("max_processing_side", 4096)))
                metrics = technical_metrics(image)
            except Exception:
                rows[input_index] = {
                    **base,
                    **asdict(QualityDecision("exclude", "missing_or_invalid_cached_image")),
                    "width": 0,
                    "height": 0,
                    "contrast_std": 0.0,
                    "entropy": 0.0,
                }
                continue
            metrics_fields = asdict(metrics)
            if source not in strict_sources:
                decision = decide_quality(
                    source, metrics, None, strict_sources=strict_sources,
                    min_image_side=int(settings["min_image_side"]),
                    min_contrast_std=float(settings["min_contrast_std"]),
                    min_entropy=float(settings["min_entropy"]),
                    min_specimen_side=int(settings["min_specimen_side"]),
                    min_specimen_area_ratio=float(settings["min_specimen_area_ratio"]),
                    crop_below_area_ratio=float(settings["crop_below_area_ratio"]),
                )
                rows[input_index] = {**base, **asdict(decision), **metrics_fields}
                image.close()
            else:
                strict_items.append((input_index, source, metrics, image))
                if len(strict_items) >= batch_size:
                    flush_strict_items()

        flush_strict_items()

        checkpoint_frame = pd.DataFrame([rows[index] for index in chunk.index])
        atomic_parquet(checkpoint, checkpoint_frame)
        print(f"quality checkpoint {chunk_index + 1}/{total_chunks}: saved", flush=True)

    decisions = pd.concat([pd.read_parquet(path) for path in checkpoint_paths], ignore_index=True)
    decisions = decisions.sort_values("input_index").reset_index(drop=True)
    if len(decisions) != len(frame) or decisions["record_id"].tolist() != frame["record_id"].astype(str).tolist():
        raise SystemExit("STOP: quality checkpoints do not align with the cached plan")

    overrides = load_overrides(overrides_path)
    unknown = sorted(set(overrides) - set(decisions["record_id"]))
    if unknown:
        raise SystemExit(f"manual overrides contain unknown record_id values: {unknown[:5]}")
    decisions["manual_override"] = ""
    for record_id, override in overrides.items():
        if not override:
            continue
        mask = decisions["record_id"].eq(record_id)
        decisions.loc[mask, "manual_override"] = override
        decisions.loc[mask, "decision"] = override
        decisions.loc[mask, "reason"] = f"manual_{override}"

    keep_mask = decisions["decision"].isin(["accept", "accept_crop"])
    accepted = frame.loc[keep_mask.to_numpy()].copy().reset_index(drop=True)
    kept_decisions = decisions.loc[keep_mask].reset_index(drop=True)
    crop_mask = kept_decisions["updated_local_path"].astype(str).ne("")
    accepted.loc[crop_mask, "local_path"] = kept_decisions.loc[crop_mask, "updated_local_path"].to_numpy()
    accepted.loc[crop_mask, "image_sha256"] = kept_decisions.loc[crop_mask, "updated_image_sha256"].to_numpy()
    accepted["eligible_supervised"] = True
    accepted["exclusion_reason"] = ""

    rejected = frame.loc[~keep_mask.to_numpy()].copy().reset_index(drop=True)
    rejected_decisions = decisions.loc[~keep_mask].reset_index(drop=True)
    rejected["eligible_supervised"] = False
    rejected["exclusion_reason"] = "quality:" + rejected_decisions["reason"].astype(str)

    atomic_parquet(out, manifest_frame(accepted))
    atomic_parquet(review_out, manifest_frame(rejected))
    atomic_parquet(decisions_out, decisions)
    preview_count = write_contact_sheet(frame, decisions, preview_out)

    by_source: dict[str, dict] = {}
    for source, group in decisions.groupby("source", dropna=False):
        counts = Counter(group["decision"].astype(str))
        total = int(len(group))
        kept = counts["accept"] + counts["accept_crop"]
        by_source[str(source)] = {
            "input": total,
            "accepted": int(kept),
            "cropped": int(counts["accept_crop"]),
            "review": int(counts["review"]),
            "excluded": int(counts["exclude"]),
            "accepted_fraction": kept / total if total else 0.0,
        }
    input_sources = set(frame["source"].astype(str))
    accepted_sources = set(accepted["source"].astype(str))
    required = input_sources if required_sources is None else set(required_sources)
    unknown_required = sorted(required - input_sources)
    if unknown_required:
        raise SystemExit(
            "STOP: required quality source(s) are absent from the cached plan: "
            + ", ".join(unknown_required)
        )
    missing_sources = sorted(input_sources - accepted_sources)
    missing_required_sources = sorted(required - accepted_sources)
    missing_optional_sources = sorted(set(missing_sources) - set(missing_required_sources))
    report = {
        "complete": not missing_required_sources,
        "input_manifest": str(manifest),
        "quality_manifest": str(out),
        "review_manifest": str(review_out),
        "decisions": str(decisions_out),
        "preview": str(preview_out) if preview_count else "",
        "manual_overrides": str(overrides_path),
        "input_rows": int(len(frame)),
        "input_manifest_sha256": file_sha256(manifest),
        "accepted_rows": int(keep_mask.sum()),
        "quality_manifest_sha256": file_sha256(out),
        "cropped_rows": int((decisions["decision"] == "accept_crop").sum()),
        "quarantined_rows": int((~keep_mask).sum()),
        "by_source": by_source,
        "by_reason": dict(Counter(decisions["reason"].astype(str))),
        "required_sources": sorted(required),
        "accepted_sources": sorted(accepted_sources),
        "missing_required_sources": missing_required_sources,
        "missing_optional_sources": missing_optional_sources,
        "settings": settings_record,
        "original_cache_preserved": True,
    }
    atomic_json(report_out, report)
    print(json.dumps({key: report[key] for key in ("input_rows", "accepted_rows", "cropped_rows", "quarantined_rows", "by_source")}, indent=2, ensure_ascii=False))
    print(f"quality-approved training manifest -> {out}")
    print(f"quarantined originals (not deleted) -> {review_out}")
    if missing_optional_sources:
        print(
            "Optional source(s) omitted because no image passed the quality gate: "
            + ", ".join(missing_optional_sources),
            flush=True,
        )
    if missing_required_sources:
        raise SystemExit(
            "STOP: quality gate kept no usable images for required source(s): "
            + ", ".join(missing_required_sources)
            + ". Diagnostics were saved; the previous approved inputs were not silently accepted."
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--review-out", required=True)
    parser.add_argument("--decisions-out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--crop-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--overrides", required=True)
    parser.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--strict-sources", default="GBIF,DiSSCo")
    parser.add_argument(
        "--required-sources", default="",
        help="Comma-separated sources that must survive the gate; default requires every input source",
    )
    parser.add_argument("--prompts", default=",".join(DEFAULT_PROMPTS))
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument("--min-image-side", type=int, default=224)
    parser.add_argument("--min-contrast-std", type=float, default=2.0)
    parser.add_argument("--min-entropy", type=float, default=1.0)
    parser.add_argument("--min-specimen-side", type=int, default=64)
    parser.add_argument("--min-specimen-area-ratio", type=float, default=0.0025)
    parser.add_argument("--crop-below-area-ratio", type=float, default=0.35)
    parser.add_argument("--crop-padding", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--checkpoint-size", type=int, default=256)
    parser.add_argument("--max-processing-side", type=int, default=4096)
    parser.add_argument("--allow-cpu", action="store_true", help="Slow diagnostic runs only")
    args = parser.parse_args()
    if not 0 < args.score_threshold < 1:
        raise SystemExit("score-threshold must be between 0 and 1")
    if not 0 <= args.min_specimen_area_ratio < args.crop_below_area_ratio <= 1:
        raise SystemExit("invalid specimen-area thresholds")
    if args.batch_size < 1 or args.checkpoint_size < 1 or args.max_processing_side < 224:
        raise SystemExit("batch/checkpoint sizes must be positive and max-processing-side at least 224")

    settings = {
        "model": args.model,
        "strict_sources": [value.strip() for value in args.strict_sources.split(",") if value.strip()],
        "prompts": [value.strip() for value in args.prompts.split(",") if value.strip()],
        "score_threshold": args.score_threshold,
        "min_image_side": args.min_image_side,
        "min_contrast_std": args.min_contrast_std,
        "min_entropy": args.min_entropy,
        "min_specimen_side": args.min_specimen_side,
        "min_specimen_area_ratio": args.min_specimen_area_ratio,
        "crop_below_area_ratio": args.crop_below_area_ratio,
        "crop_padding": args.crop_padding,
        "batch_size": args.batch_size,
        "checkpoint_size": args.checkpoint_size,
        "max_processing_side": args.max_processing_side,
    }
    run_filter(
        manifest=Path(args.manifest), out=Path(args.out), review_out=Path(args.review_out),
        decisions_out=Path(args.decisions_out), report_out=Path(args.report),
        preview_out=Path(args.preview), crop_root=Path(args.crop_root),
        checkpoint_dir=Path(args.checkpoint_dir), overrides_path=Path(args.overrides),
        settings=settings, allow_cpu=args.allow_cpu,
        required_sources=(
            {value.strip() for value in args.required_sources.split(",") if value.strip()}
            if args.required_sources.strip() else None
        ),
    )


if __name__ == "__main__":
    main()
