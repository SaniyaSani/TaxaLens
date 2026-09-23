#!/usr/bin/env python3
"""Download and safely extract the official monthly iNaturalist metadata bundle."""
from __future__ import annotations

import argparse
import gzip
import shutil
import tarfile
from pathlib import Path

DEFAULT_URL = "https://inaturalist-open-data.s3.amazonaws.com/metadata/inaturalist-open-data-latest.tar.gz"
WANTED = {"observations.csv.gz", "photos.csv.gz", "taxa.csv.gz", "observers.csv.gz"}
PLAIN_TO_GZIP = {name.removesuffix(".gz"): name for name in WANTED}


def safe_member_name(member: tarfile.TarInfo) -> str | None:
    name = Path(member.name).name
    if not member.isfile():
        return None
    if name in WANTED:
        return name
    return PLAIN_TO_GZIP.get(name)


def download(url: str, destination: Path) -> None:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        # Keep the object bytes exactly as stored by S3.  Some HTTP clients
        # otherwise decode Content-Encoding: gzip transparently, leaving a
        # perfectly valid plain TAR behind a misleading .tar.gz suffix.
        response.raw.decode_content = False
        with partial.open("wb") as handle:
            shutil.copyfileobj(response.raw, handle, length=8 * 1024 * 1024)
    partial.replace(destination)


def extract_selected(archive: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    # ``r:*`` accepts both the documented gzip bundle and a plain TAR produced
    # when an HTTP client has already decoded the transport-level gzip stream.
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle:
            name = safe_member_name(member)
            if not name:
                continue
            source = bundle.extractfile(member)
            if source is None:
                continue
            target = out_dir / name
            partial = target.with_suffix(target.suffix + ".part")
            partial.unlink(missing_ok=True)
            try:
                with source, partial.open("wb") as raw_destination:
                    if Path(member.name).name.endswith(".gz"):
                        shutil.copyfileobj(
                            source, raw_destination, length=8 * 1024 * 1024
                        )
                    else:
                        # Downstream stages intentionally use the stable
                        # ``*.csv.gz`` paths.  Compress plain CSV members while
                        # streaming so the 90+ GB bundle is never duplicated in
                        # memory or extracted as a second uncompressed copy.
                        with gzip.GzipFile(
                            filename="",
                            mode="wb",
                            compresslevel=1,
                            fileobj=raw_destination,
                            mtime=0,
                        ) as destination:
                            shutil.copyfileobj(
                                source, destination, length=8 * 1024 * 1024
                            )
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
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
