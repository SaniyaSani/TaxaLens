from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from scripts.filter_image_quality import (
    Detection,
    TechnicalMetrics,
    decide_quality,
    run_filter,
)


ROOT = Path(__file__).resolve().parents[1]


STRICT = {"GBIF", "DiSSCo"}


def decide(source: str, detection: Detection | None):
    return decide_quality(
        source,
        TechnicalMetrics(width=1000, height=800, contrast_std=30, entropy=5),
        detection,
        strict_sources=STRICT,
        min_image_side=224,
        min_contrast_std=2,
        min_entropy=1,
        min_specimen_side=64,
        min_specimen_area_ratio=0.0025,
        crop_below_area_ratio=0.35,
    )


def test_museum_overview_without_visible_fly_is_quarantined():
    result = decide("DiSSCo", None)
    assert result.decision == "review"
    assert result.reason == "no_fly_detected"


def test_tiny_museum_specimen_is_quarantined_but_detailed_one_is_cropped():
    tiny = decide("GBIF", Detection("a pinned fly", 0.7, (100, 100, 130, 125)))
    assert tiny.decision == "review" and tiny.reason == "specimen_too_small"
    detailed = decide("GBIF", Detection("a pinned fly", 0.7, (200, 150, 600, 550)))
    assert detailed.decision == "accept_crop"
    assert detailed.specimen_short_side_px == 400


def test_standardized_and_field_sources_do_not_get_museum_false_rejects():
    assert decide("BIOSCAN-5M", None).decision == "accept"
    assert decide("iNaturalist", None).decision == "accept"


class FakeDetector:
    def __init__(self, detections):
        self.detections = list(detections)
        self.calls = 0
        self.max_batch_size = 0

    def detect(self, images):
        self.calls += 1
        self.max_batch_size = max(self.max_batch_size, len(images))
        result = self.detections[: len(images)]
        self.detections = self.detections[len(images) :]
        return result


def specimen_image(path: Path, color: str) -> None:
    image = Image.new("RGB", (600, 400), "#eeeeee")
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 80, 500, 320), fill=color)
    draw.line((100, 80, 500, 320), fill="black", width=8)
    image.save(path)


def settings():
    return {
        "model": "fake/owlv2",
        "strict_sources": ["GBIF", "DiSSCo"],
        "prompts": ["a fly"],
        "score_threshold": 0.1,
        "min_image_side": 224,
        "min_contrast_std": 2.0,
        "min_entropy": 1.0,
        "min_specimen_side": 64,
        "min_specimen_area_ratio": 0.0025,
        "crop_below_area_ratio": 0.35,
        "crop_padding": 0.5,
        "batch_size": 2,
        "checkpoint_size": 10,
    }


def paths(root: Path):
    return {
        "manifest": root / "cached.parquet",
        "out": root / "quality.parquet",
        "review_out": root / "review.parquet",
        "decisions_out": root / "decisions.parquet",
        "report_out": root / "report.json",
        "preview_out": root / "preview.jpg",
        "crop_root": root / "crops",
        "checkpoint_dir": root / "checkpoints",
        "overrides_path": root / "overrides.csv",
    }


def test_filter_preserves_originals_crops_good_museum_media_and_resumes(tmp_path):
    rows = []
    sources = ["BIOSCAN-5M", "iNaturalist", "GBIF", "DiSSCo", "DiSSCo"]
    for index, source in enumerate(sources):
        image_path = tmp_path / f"original-{index}.jpg"
        specimen_image(image_path, f"#{40 + index * 30:02x}6080")
        rows.append({
            "record_id": f"img:{index}",
            "source": source,
            "source_record_id": str(index),
            "source_image_id": str(index),
            "local_path": str(image_path),
            "image_url": f"https://example.org/{index}.jpg",
            "image_license": "CC-BY",
            "order": "Diptera",
            "family": "Muscidae",
            "eligible_supervised": True,
        })
    target = paths(tmp_path)
    pd.DataFrame(rows).to_parquet(target["manifest"], index=False)
    originals = {row["local_path"]: Path(row["local_path"]).read_bytes() for row in rows}
    detector = FakeDetector([
        Detection("a fly", 0.8, (180, 100, 420, 300)),
        Detection("a pinned fly", 0.7, (160, 80, 440, 320)),
        Detection("a pinned fly", 0.9, (20, 20, 45, 40)),
    ])

    report = run_filter(**target, settings=settings(), detector=detector)
    approved = pd.read_parquet(target["out"])
    review = pd.read_parquet(target["review_out"])
    assert report["input_rows"] == 5
    assert report["accepted_rows"] == 4
    assert report["cropped_rows"] == 2
    assert review["record_id"].tolist() == ["img:4"]
    assert set(approved["source"]) == set(sources)
    assert detector.max_batch_size <= settings()["batch_size"]
    assert all(Path(path).read_bytes() == content for path, content in originals.items())
    assert target["preview_out"].is_file()

    # A second run must use the saved checkpoint and need no model inference.
    no_calls = FakeDetector([])
    run_filter(**target, settings=settings(), detector=no_calls)
    assert no_calls.calls == 0

    # Manual review can explicitly restore one quarantined record without
    # re-running the detector or modifying the original JPEG.
    target["overrides_path"].write_text("record_id,decision\nimg:4,accept\n")
    report = run_filter(**target, settings=settings(), detector=no_calls)
    assert report["accepted_rows"] == 5
    assert len(pd.read_parquet(target["out"])) == 5


def test_optional_source_with_no_usable_images_is_reported_not_forced_into_training(tmp_path):
    rows = []
    for index, source in enumerate(["BIOSCAN-5M", "DiSSCo"]):
        image_path = tmp_path / f"optional-{index}.jpg"
        specimen_image(image_path, f"#{80 + index * 30:02x}6080")
        rows.append({
            "record_id": f"optional:{index}",
            "source": source,
            "source_record_id": str(index),
            "source_image_id": str(index),
            "local_path": str(image_path),
            "image_url": f"https://example.org/optional/{index}.jpg",
            "image_license": "CC-BY",
            "order": "Diptera",
            "family": "Muscidae",
            "eligible_supervised": True,
        })
    target = paths(tmp_path)
    pd.DataFrame(rows).to_parquet(target["manifest"], index=False)

    report = run_filter(
        **target, settings=settings(), detector=FakeDetector([None]),
        required_sources={"BIOSCAN-5M"},
    )

    assert report["complete"] is True
    assert report["missing_required_sources"] == []
    assert report["missing_optional_sources"] == ["DiSSCo"]
    assert pd.read_parquet(target["out"])["source"].tolist() == ["BIOSCAN-5M"]
    assert pd.read_parquet(target["review_out"])["source"].tolist() == ["DiSSCo"]

    # Completed checkpoints can be finalized without constructing a GPU model.
    resumed = run_filter(
        **target, settings=settings(), detector=None,
        required_sources={"BIOSCAN-5M"},
    )
    assert resumed["complete"] is True


def test_quality_shard_refresh_removes_only_stale_derived_manifests(tmp_path):
    source = tmp_path / "quality.parquet"
    out = tmp_path / "shards"
    frame = pd.DataFrame({
        "record_id": [f"img:{index}" for index in range(21)],
        "source": ["BIOSCAN-5M"] * 21,
        "specimen_group_id": [f"group:{index}" for index in range(21)],
    })
    frame.to_parquet(source, index=False)
    command = [sys.executable, str(ROOT / "scripts/make_embedding_shards.py"),
               "--input", str(source), "--out-dir", str(out), "--shard-size", "10"]
    subprocess.run(command, cwd=ROOT, check=True)
    assert len(list(out.glob("shard_*.parquet"))) == 3

    frame.iloc[:5].to_parquet(source, index=False)
    subprocess.run(command, cwd=ROOT, check=True)
    assert [path.name for path in out.glob("shard_*.parquet")] == ["shard_00000.parquet"]
    index = json.loads((out / "shards.json").read_text())
    assert index["shard_count"] == 1 and len(index["input_sha256"]) == 64


def test_merge_uses_only_current_quality_manifest_shards(tmp_path):
    manifest_root = tmp_path / "manifests"
    embedded_root = tmp_path / "embedded"
    output = tmp_path / "model"
    manifest_root.mkdir()
    current_manifest = manifest_root / "shard_00000.parquet"
    pd.DataFrame([{"record_id": "current"}]).to_parquet(current_manifest, index=False)
    for name, record_id, value in [
        ("shard_00000", "current", 1.0),
        ("shard_00001", "stale", 9.0),
    ]:
        directory = embedded_root / name
        directory.mkdir(parents=True)
        np.save(directory / "embeddings.npy", np.array([[value, value]], dtype=np.float32))
        pd.DataFrame([{"record_id": record_id}]).to_csv(directory / "embedded_manifest.csv", index=False)
        vector_path = directory / "embeddings.npy"
        frame_path = directory / "embedded_manifest.csv"
        (directory / "complete.json").write_text(json.dumps({
            "manifest_sha256": hashlib.sha256(
                (current_manifest if name == "shard_00000" else frame_path).read_bytes()
            ).hexdigest(),
            "embedding": {"backend": "dino", "backbone": "fake", "image_size": 512,
                          "tile_grid": 1, "include_whole": True, "local_only": False},
            "artifact_sha256": {
                "embeddings.npy": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
                "embedded_manifest.csv": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
            },
        }))
    subprocess.run([
        sys.executable, str(ROOT / "scripts/merge_embedding_shards.py"),
        "--shard-root", str(embedded_root), "--manifest-root", str(manifest_root),
        "--out-dir", str(output),
    ], cwd=ROOT, check=True)
    merged = np.load(output / "embeddings.npy")
    assert merged.shape == (1, 2) and merged[0, 0] == 1.0
    assert pd.read_csv(output / "embedded_manifest.csv")["record_id"].tolist() == ["current"]
