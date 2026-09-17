#!/usr/bin/env python3
"""Download and safely extract the official monthly iNaturalist metadata bundle."""
from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

DEFAULT_URL = "https://inaturalist-open-data.s3.amazonaws.com/metadata/inaturalist-open-data-latest.tar.gz"
WANTED = {"observations.csv.gz", "photos.csv.gz", "taxa.csv.gz", "observers.csv.gz"}


def safe_member_name(member: tarfile.TarInfo) -> str | None:
    name = Path(member.name).name
    return name if member.isfile() and name in WANTED else None


def download(url: str, destination: Path) -> None:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)


def extract_selected(archive: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            name = safe_member_name(member)
            if not name:
                continue
            source = bundle.extractfile(member)
            if source is None:
                continue
            target = out_dir / name
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            extracted.append(target)
    missing = WANTED - {path.name for path in extracted}
    if missing:
        raise SystemExit(f"metadata archive is missing: {', '.join(sorted(missing))}")
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/raw/inaturalist")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--archive", help="Use an already downloaded tar.gz instead")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--allow-bulk-download", action="store_true", help="Explicitly allow the complete worldwide metadata archive; NOT needed for bounded PoC")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    archive = Path(args.archive) if args.archive else out_dir / "inaturalist-open-data-latest.tar.gz"
    if not args.archive:
        if archive.exists() and not args.force:
            print(f"reuse existing archive: {archive}")
        else:
            if not args.allow_bulk_download:
                raise SystemExit("STOP: this is the complete worldwide iNaturalist metadata bundle. For PoC use run_multisource_poc_v09.py --stage download --source inat --reuse-existing-bioscan. Bulk mode requires explicit --allow-bulk-download after checking storage.")
            print(f"downloading {args.url}")
            download(args.url, archive)
    paths = extract_selected(archive, out_dir)
    for path in paths:
        print(path)
    if not args.keep_archive and not args.archive:
        archive.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
