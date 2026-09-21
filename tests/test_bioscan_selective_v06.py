from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

from scripts.download_bioscan_subset import member_matches


ROOT = Path(__file__).resolve().parents[1]


def run_script(name: str, *args: object) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *(str(arg) for arg in args)],
        cwd=ROOT,
        check=True,
    )


def jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_member_matching_tolerates_official_archive_prefix():
    assert member_matches(
        "bioscan5m/images/cropped_256/train/a/ABC-1.jpg",
        "train/a/ABC-1.jpg",
    )
    assert not member_matches("train/a/ABC-2.jpg", "train/a/ABC-1.jpg")


def test_metadata_first_selector_is_exact_balanced_and_deterministic(tmp_path: Path):
    rows = []
    splits = ["pretrain", "train", "val", "test"]
    families = ["Phoridae", "Cecidomyiidae", "Sciaridae"]
    for index in range(80):
        split = splits[index % len(splits)]
        family = families[index % len(families)]
        rows.append({
            "processid": f"BIO-{index:03d}",
            "order": "Diptera",
            "family": family,
            "genus": f"Genus{index % 9}",
            "species": f"Genus{index % 9} species{index % 17}" if index % 2 else "sp.",
            "dna_barcode": "ACGT" * 10,
            "dna_bin": f"BOLD:{index % 23:04d}",
            "split": split,
            "chunk": f"{index % 16:02x}" if split in {"pretrain", "train"} else "",
        })
    rows.append({
        "processid": "NOT-A-FLY", "order": "Lepidoptera", "family": "Noctuidae",
        "genus": "Noctua", "species": "Noctua pronuba", "split": "train", "chunk": "0",
    })
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata, index=False)

    targets = "pretrain=6,train=6,val=3,test=3"
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    for out in (first, second):
        run_script(
            "select_bioscan_diptera.py",
            "--metadata", metadata,
            "--out", out,
            "--report", out.with_suffix(".json"),
            "--image-dir", tmp_path / "images",
            "--max-records", 18,
            "--max-per-taxon", 10,
            "--split-targets", targets,
            "--chunksize", 13,
        )

    selected = pd.read_csv(first, dtype=str, keep_default_na=False)
    repeated = pd.read_csv(second, dtype=str, keep_default_na=False)
    assert len(selected) == 18
    assert set(selected["order"]) == {"Diptera"}
    assert selected["processid"].tolist() == repeated["processid"].tolist()
    assert selected.groupby("source_split").size().to_dict() == {
        "pretrain": 6, "train": 6, "val": 3, "test": 3,
    }
    assert all(member.endswith(f"{process_id}.jpg") for member, process_id in zip(selected["archive_member"], selected["processid"]))
    assert "NOT-A-FLY" not in set(selected["processid"])


def test_metadata_selector_restricts_bioscan_to_versioned_family_scope(tmp_path: Path):
    rows = []
    for index, family in enumerate(("Phoridae", "Sciaridae", "Syrphidae") * 8):
        rows.append({
            "processid": f"SCOPE-{index:03d}",
            "order": "Diptera",
            "family": family,
            "genus": f"Genus{index}",
            "species": f"Genus{index} species",
            "split": "train",
            "chunk": "00",
        })
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata, index=False)
    family_scope = tmp_path / "families.json"
    family_scope.write_text(
        json.dumps({"families": ["Phoridae", "Syrphidae"]}),
        encoding="utf-8",
    )
    selected_path = tmp_path / "selected.csv"
    report_path = tmp_path / "report.json"

    run_script(
        "select_bioscan_diptera.py",
        "--metadata", metadata,
        "--out", selected_path,
        "--report", report_path,
        "--image-dir", tmp_path / "images",
        "--max-records", 8,
        "--max-per-taxon", 8,
        "--split-targets", "train=8",
        "--families-file", family_scope,
        "--chunksize", 7,
    )

    selected = pd.read_csv(selected_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(selected) == 8
    assert set(selected["family"]) <= {"Phoridae", "Syrphidae"}
    assert "Sciaridae" not in set(selected["family"])
    assert report["target_families"] == ["Phoridae", "Syrphidae"]


def test_selective_downloader_reads_only_chosen_local_zip_members_and_resumes(tmp_path: Path):
    selection_rows = [
        {"processid": "P-1", "archive_group": "pretrain", "archive_member": "pretrain/aa/P-1.jpg"},
        {"processid": "T-1", "archive_group": "train", "archive_member": "train/b/T-1.jpg"},
        {"processid": "E-1", "archive_group": "eval", "archive_member": "val/E-1.jpg"},
    ]
    selection = tmp_path / "selection.csv"
    pd.DataFrame(selection_rows).to_csv(selection, index=False)

    archives = {}
    for group, member, color in (
        ("pretrain", "prefix/pretrain/aa/P-1.jpg", (255, 0, 0)),
        ("train", "prefix/train/b/T-1.jpg", (0, 255, 0)),
        ("eval", "prefix/val/E-1.jpg", (0, 0, 255)),
    ):
        archive = tmp_path / f"{group}.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as handle:
            handle.writestr(member, jpeg_bytes(color))
            handle.writestr(f"unselected/{group}.jpg", jpeg_bytes((1, 2, 3)))
        archives[group] = [{"name": group, "path": str(archive)}]
    config = tmp_path / "archives.json"
    config.write_text(json.dumps({"archives": archives}), encoding="utf-8")

    images = tmp_path / "images"
    completed = tmp_path / "completed.csv"
    report = tmp_path / "report.json"
    command = (
        "--selection", selection,
        "--archive-config", config,
        "--image-dir", images,
        "--out-manifest", completed,
        "--report", report,
        "--checkpoint-every", 1,
    )
    run_script("download_bioscan_subset.py", *command)
    assert len(pd.read_csv(completed)) == 3
    assert {path.name for path in images.glob("*.jpg")} == {"P-1.jpg", "T-1.jpg", "E-1.jpg"}
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["complete"] == 3
    assert payload["downloaded_now"] == 3
    assert payload["full_archives_downloaded"] is False

    # The second run validates and skips all existing JPEGs.
    run_script("download_bioscan_subset.py", *command)
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["downloaded_now"] == 0
    assert payload["skipped_existing"] == 3
