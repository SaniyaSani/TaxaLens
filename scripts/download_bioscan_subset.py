#!/usr/bin/env python3
"""Download only selected BIOSCAN-5M JPEG members from the official ZIP files.

For HTTP(S) archives this script uses byte-range requests through ``remotezip``.
It reads the ZIP directory and the compressed bytes of requested members only;
the 36+ GB cropped-image corpus is never downloaded as a whole.  Existing valid
JPEGs are skipped, so the command is safe to stop and resume.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Iterator

import pandas as pd
from PIL import Image


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "bioscan_archives_v06.json"
USER_AGENT = "TaxaLens-BIOSCAN-Selective/0.6 (licensed biodiversity research)"


def is_remote(location: str) -> bool:
    return location.startswith(("http://", "https://"))


def normalized_member(value: str) -> str:
    value = value.replace("\\", "/").lstrip("./")
    while "//" in value:
        value = value.replace("//", "/")
    return value


def member_matches(candidate: str, wanted: str) -> bool:
    candidate = normalized_member(candidate)
    wanted = normalized_member(wanted)
    return candidate == wanted or candidate.endswith("/" + wanted)


def valid_image_bytes(data: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        return True
    except Exception:
        return False


def valid_existing(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


@contextmanager
def open_archive(location: str, timeout: int, initial_buffer_mb: int, suffix_range: bool):
    """Open a local ZIP or a remote range-backed ZIP with a common interface."""
    if is_remote(location):
        try:
            from remotezip import RemoteZip
        except ImportError as exc:
            raise SystemExit("Selective BIOSCAN download requires: pip install remotezip") from exc
        archive = RemoteZip(
            location,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            initial_buffer_size=initial_buffer_mb * 1024 * 1024,
            support_suffix_range=suffix_range,
        )
    else:
        archive = zipfile.ZipFile(Path(location).expanduser())
    try:
        yield archive
    finally:
        archive.close()


def index_wanted_members(archive, rows: list[dict]) -> dict[int, str]:
    """Match selected relative member paths despite archive-specific prefixes."""
    wanted_by_basename: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        wanted = normalized_member(str(row["archive_member"]))
        wanted_by_basename[PurePosixPath(wanted).name].append((row_index, wanted))

    matches: dict[int, str] = {}
    for info in archive.infolist():
        candidate = normalized_member(info.filename)
        if info.is_dir():
            continue
        options = wanted_by_basename.get(PurePosixPath(candidate).name, ())
        for row_index, wanted in options:
            if row_index not in matches and member_matches(candidate, wanted):
                matches[row_index] = info.filename
    return matches


def atomic_write_image(path: Path, data: bytes) -> None:
    if not valid_image_bytes(data):
        raise ValueError("downloaded member is not a valid image")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def archive_entries(config: dict, group: str) -> Iterator[dict]:
    entries = config.get("archives", {}).get(group, [])
    if not entries:
        raise SystemExit(f"archive config has no entries for group: {group}")
    yield from entries


def output_path(row: dict, override_dir: Path | None) -> Path:
    process_id = str(row.get("processid") or row.get("process_id") or row.get("sampleid") or "").strip()
    if not process_id:
        raise ValueError("selection row has no processid")
    if override_dir is not None:
        return (override_dir / f"{process_id}.jpg").resolve()
    local_path = str(row.get("local_path", "")).strip()
    if not local_path:
        raise ValueError("selection row has no local_path; pass --image-dir")
    return Path(local_path).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, help="CSV from select_bioscan_diptera.py")
    parser.add_argument("--archive-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--image-dir", help="Override output directory stored in the selection")
    parser.add_argument("--out-manifest", help="Completed selection CSV; default: <selection>_downloaded.csv")
    parser.add_argument("--report", help="Download report JSON")
    parser.add_argument("--progress", help="Resumable progress JSON")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--initial-buffer-mb", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--limit", type=int, default=0, help="Testing only: process the first N selection rows")
    parser.add_argument("--no-suffix-range", action="store_true", help="Use HEAD + absolute ranges for servers without suffix ranges")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    selection_path = Path(args.selection)
    frame = pd.read_csv(selection_path, dtype=str, keep_default_na=False)
    required = {"processid", "archive_group", "archive_member"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise SystemExit(f"selection is missing columns: {', '.join(sorted(missing_columns))}")
    if args.limit:
        frame = frame.head(args.limit).copy()
    if frame.empty:
        raise SystemExit("BIOSCAN selection is empty")

    config = json.loads(Path(args.archive_config).read_text(encoding="utf-8"))
    override_dir = Path(args.image_dir).expanduser() if args.image_dir else None
    out_manifest = Path(args.out_manifest) if args.out_manifest else selection_path.with_name(selection_path.stem + "_downloaded.csv")
    report_path = Path(args.report) if args.report else selection_path.with_name(selection_path.stem + "_download_report.json")
    progress_path = Path(args.progress) if args.progress else selection_path.with_name(selection_path.stem + "_progress.json")

    rows = frame.to_dict(orient="records")
    paths: dict[int, Path] = {}
    complete: set[int] = set()
    errors: list[dict] = []
    downloaded_now = 0
    skipped_existing = 0
    downloaded_bytes = 0
    archive_indexed: list[str] = []

    for index, row in enumerate(rows):
        path = output_path(row, override_dir)
        paths[index] = path
        rows[index]["local_path"] = str(path)
        if valid_existing(path):
            complete.add(index)
            skipped_existing += 1
        # Invalid files are replaced only after a new JPEG has been verified.

    def checkpoint(active_archive: str = "") -> None:
        save_progress(progress_path, {
            "requested": len(rows),
            "complete": len(complete),
            "downloaded_now": downloaded_now,
            "skipped_existing": skipped_existing,
            "downloaded_bytes": downloaded_bytes,
            "active_archive": active_archive,
            "errors": errors[-100:],
            "full_archives_downloaded": False,
        })

    checkpoint()
    for group in ("train", "eval", "pretrain"):
        group_indices = [index for index, row in enumerate(rows) if row["archive_group"] == group and index not in complete]
        if not group_indices:
            continue
        for entry in archive_entries(config, group):
            remaining = [index for index in group_indices if index not in complete]
            if not remaining:
                break
            location = str(entry.get("path") or entry.get("url") or "")
            archive_name = str(entry.get("name") or Path(location).name)
            if not location:
                raise SystemExit(f"archive entry {archive_name!r} has neither path nor url")
            print(f"Indexing {archive_name}; full ZIP will not be downloaded...")
            try:
                with open_archive(
                    location,
                    timeout=args.timeout,
                    initial_buffer_mb=args.initial_buffer_mb,
                    suffix_range=not args.no_suffix_range,
                ) as archive:
                    local_rows = [rows[index] for index in remaining]
                    matches = index_wanted_members(archive, local_rows)
                    archive_indexed.append(archive_name)
                    for local_index, member in matches.items():
                        global_index = remaining[local_index]
                        if global_index in complete:
                            continue
                        last_error: Exception | None = None
                        for attempt in range(args.retries + 1):
                            try:
                                data = archive.read(member)
                                atomic_write_image(paths[global_index], data)
                                downloaded_bytes += len(data)
                                downloaded_now += 1
                                complete.add(global_index)
                                last_error = None
                                break
                            except Exception as exc:
                                last_error = exc
                                if attempt < args.retries:
                                    time.sleep(1.5 * (attempt + 1))
                        if last_error is not None:
                            errors.append({
                                "processid": rows[global_index]["processid"],
                                "archive": archive_name,
                                "member": member,
                                "error": f"{type(last_error).__name__}: {last_error}",
                            })
                        if (downloaded_now + len(errors)) % args.checkpoint_every == 0:
                            checkpoint(archive_name)
                            print(f"BIOSCAN images complete: {len(complete):,}/{len(rows):,}", end="\r")
            except Exception as exc:
                errors.append({"archive": archive_name, "error": f"{type(exc).__name__}: {exc}"})
                checkpoint(archive_name)
                print(f"Could not read {archive_name}: {exc}")
        unresolved = [index for index in group_indices if index not in complete]
        for index in unresolved:
            if not any(error.get("processid") == rows[index]["processid"] for error in errors):
                errors.append({
                    "processid": rows[index]["processid"],
                    "archive_group": group,
                    "member": rows[index]["archive_member"],
                    "error": "member not found in configured archives",
                })
        checkpoint()

    completed_rows = [rows[index] for index in sorted(complete)]
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    pending_manifest = out_manifest.with_name(out_manifest.name + ".pending")
    pd.DataFrame(completed_rows, columns=frame.columns.union(["local_path"], sort=False)).to_csv(pending_manifest, index=False)
    pending_manifest.replace(out_manifest)
    completed_by_group = Counter(rows[index]["archive_group"] for index in complete)
    report = {
        "requested": len(rows),
        "complete": len(complete),
        "downloaded_now": downloaded_now,
        "skipped_existing": skipped_existing,
        "downloaded_bytes": downloaded_bytes,
        "complete_by_archive_group": dict(completed_by_group),
        "archives_indexed": archive_indexed,
        "errors": errors,
        "full_archives_downloaded": False,
        "range_download": True,
        "output_manifest": str(out_manifest),
    }
    save_progress(report_path, report)
    checkpoint()
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}, indent=2))
    print(f"Downloaded BIOSCAN selection -> {out_manifest}")
    if len(complete) != len(rows) and not args.allow_partial:
        raise SystemExit(f"BIOSCAN download incomplete: {len(complete):,}/{len(rows):,}; rerun to resume")


if __name__ == "__main__":
    main()
