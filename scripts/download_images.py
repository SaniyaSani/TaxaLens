#!/usr/bin/env python3
"""Download licensed manifest images into a deterministic local cache."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import re
import sys
import threading
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diptera_id.corpus.io import ManifestWriter, iter_table
from diptera_id.corpus.schema import finalize_record, normalize_license


UA = "TaxaLens-Foundation/0.5 (licensed research image cache)"
_THREAD_LOCAL = threading.local()
_RATE_LOCK = threading.Lock()
_NEXT_REQUEST_AT = 0.0


def safe_source(value: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-") or "unknown"


def session_for_thread() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": UA})
        _THREAD_LOCAL.session = session
    return session


def wait_for_request_slot(interval: float) -> None:
    """Keep one global request-start rate while allowing slow requests to overlap."""
    if interval <= 0:
        return

    global _NEXT_REQUEST_AT
    with _RATE_LOCK:
        now = time.monotonic()
        delay = max(0.0, _NEXT_REQUEST_AT - now)
        _NEXT_REQUEST_AT = max(now, _NEXT_REQUEST_AT) + interval

    if delay:
        time.sleep(delay)


def cache_record(
    raw: dict,
    allowed: set[str],
    root: Path,
    request_interval: float,
    connect_timeout: float = 45.0,
    read_timeout: float = 45.0,
    retries: int = 0,
    retry_backoff: float = 2.0,
) -> tuple[dict, int, int]:
    record = finalize_record(raw)

    if record["image_license"] not in allowed:
        record["eligible_supervised"] = False
        record["exclusion_reason"] = f"license:{record['image_license']}"
        return record, 0, 0

    if record["local_path"] and Path(record["local_path"]).is_file():
        return record, 0, 0

    if not record["image_url"]:
        record["eligible_supervised"] = False
        record["exclusion_reason"] = "missing_image_location"
        return record, 0, 0

    destination = (
        root
        / safe_source(record["source"])
        / record["record_id"][:2]
        / f"{record['record_id']}.jpg"
    )

    if destination.exists():
        try:
            with Image.open(destination) as existing:
                existing.verify()
            record["local_path"] = str(destination)
            record["image_sha256"] = hashlib.sha256(
                destination.read_bytes()
            ).hexdigest()
            return record, 0, 0
        except Exception:
            destination.unlink(missing_ok=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    try:
        content = None
        retryable_statuses = {
            408,
            425,
            429,
            500,
            502,
            503,
            504,
        }

        for attempt in range(retries + 1):
            try:
                wait_for_request_slot(request_interval)

                with session_for_thread().get(
                    record["image_url"],
                    timeout=(connect_timeout, read_timeout),
                ) as response:
                    if (
                        response.status_code in retryable_statuses
                        and attempt < retries
                    ):
                        response.close()
                        time.sleep(
                            retry_backoff * (2 ** attempt)
                        )
                        continue

                    response.raise_for_status()
                    content = response.content

                break

            except (
                requests.Timeout,
                requests.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
            ):
                if attempt >= retries:
                    raise

                time.sleep(
                    retry_backoff * (2 ** attempt)
                )

        if content is None:
            raise RuntimeError(
                "download attempts ended without image content"
            )

        with Image.open(BytesIO(content)) as opened:
            image = opened.convert("RGB")

        image.save(temporary, "JPEG", quality=94)
        temporary.replace(destination)

        record["local_path"] = str(destination)
        record["image_sha256"] = hashlib.sha256(
            destination.read_bytes()
        ).hexdigest()
        return record, 1, 1

    except Exception as exc:
        temporary.unlink(missing_ok=True)
        record["eligible_supervised"] = False
        record["exclusion_reason"] = f"download_error:{type(exc).__name__}"
        return record, 1, 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--out",
        default="data/corpus/downloaded_manifest.parquet",
    )
    parser.add_argument("--image-root", default="data/images/corpus")
    parser.add_argument("--licenses", default="CC0,CC-BY,CC-BY-SA")
    parser.add_argument(
        "--max-images",
        type=int,
        default=1_000,
        help="Pilot safety cap; 0 requires --allow-unbounded",
    )
    parser.add_argument(
        "--allow-unbounded",
        action="store_true",
        help="Acknowledge storage, bandwidth and source-policy responsibility",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Minimum interval between request starts across all workers",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=45.0,
        help="Connection timeout in seconds",
    )
    parser.add_argument(
        "--read-timeout",
        type=float,
        default=45.0,
        help="Image response timeout in seconds",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Retries after transient network failures",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=2.0,
        help="Initial exponential retry delay in seconds",
    )
    parser.add_argument("--chunksize", type=int, default=1_000)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if args.max_images == 0 and not args.allow_unbounded:
        raise SystemExit(
            "Use --allow-unbounded together with --max-images 0 "
            "after reviewing source terms and storage"
        )
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.connect_timeout <= 0:
        raise SystemExit("--connect-timeout must be positive")
    if args.read_timeout <= 0:
        raise SystemExit("--read-timeout must be positive")
    if args.retries < 0:
        raise SystemExit("--retries cannot be negative")
    if args.retry_backoff < 0:
        raise SystemExit("--retry-backoff cannot be negative")

    allowed = {
        normalize_license(value)
        for value in args.licenses.split(",")
    }
    root = Path(args.image_root)
    attempted = 0
    downloaded = 0

    with ManifestWriter(args.out) as writer:
        stop = False

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for chunk in iter_table(args.manifest, args.chunksize):
                raw_records = chunk.to_dict(orient="records")
                output = []

                if args.max_images:
                    for raw in raw_records:
                        record, tried, saved = cache_record(
                            raw,
                            allowed,
                            root,
                            args.sleep,
                        )
                        output.append(record)
                        attempted += tried
                        downloaded += saved

                        if attempted >= args.max_images:
                            stop = True
                            break
                else:
                    results = pool.map(
                        lambda raw: cache_record(
                            raw,
                            allowed,
                            root,
                            args.sleep,
                        ),
                        raw_records,
                    )
                    for record, tried, saved in results:
                        output.append(record)
                        attempted += tried
                        downloaded += saved

                writer.write(output)
                print(
                    f"rows: {writer.rows_written} | "
                    f"attempted: {attempted} | downloaded: {downloaded}",
                    flush=True,
                )

                if stop:
                    break

    print(
        f"downloaded {downloaded}/{attempted}; "
        f"updated manifest -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
