#!/usr/bin/env python3
"""Run a resumable BIOSCAN-5M Diptera workflow at a configurable scale.

Unlike the legacy ``run_bioscan_30k.py`` entry point, every generated path is
derived from ``--dataset-tag``.  This keeps independent 60k, 150k and 300k
experiments from silently reusing one another's selections or checkpoints.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from select_bioscan_diptera import load_family_filter

ROOT = Path(__file__).resolve().parents[1]
SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def run(*args: str) -> None:
    print("+", shlex.join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def count_tag(max_records: int) -> str:
    """Return a compact, stable filename tag for a positive record count."""
    if max_records < 1:
        raise ValueError("max-records must be positive")
    if max_records % 1_000_000 == 0:
        return f"{max_records // 1_000_000}m"
    if max_records % 1_000 == 0:
        return f"{max_records // 1_000}k"
    return str(max_records)


def validate_tag(value: str) -> str:
    tag = str(value).strip()
    if not SAFE_TAG.fullmatch(tag) or tag in {".", ".."}:
        raise ValueError(
            "dataset-tag must be one safe filename component "
            "(letters, digits, dot, underscore or hyphen)"
        )
    return tag


def dataset_paths(root: str | Path, dataset_tag: str) -> dict[str, Path]:
    store = Path(root).expanduser().resolve()
    tag = validate_tag(dataset_tag)
    prefix = f"diptera_{tag}"
    return {
        "root": store,
        "selection": store / f"{prefix}_selection.csv",
        "selection_report": store / f"{prefix}_selection_report.json",
        "images": store / f"{prefix}_images",
        "downloaded": store / f"{prefix}_downloaded.csv",
        "download_report": store / f"{prefix}_download_report.json",
        "download_progress": store / f"{prefix}_progress.json",
        "existing_report": store / f"{prefix}_existing_report.json",
        "manifest": store / f"bioscan_{prefix}_manifest.parquet",
    }


def find_metadata(root: Path) -> Path:
    preferred = root / "bioscan5m" / "metadata" / "csv" / "BIOSCAN_5M_Insect_Dataset_metadata.csv"
    if preferred.exists():
        return preferred
    candidates = sorted(root.rglob("*metadata*.csv"))
    if not candidates:
        raise SystemExit(f"BIOSCAN metadata was not found under {root}")
    return candidates[0]


def verify_existing_selection(
    selection: Path,
    report_path: Path,
    *,
    max_records: int,
    max_per_taxon: int,
    min_rank: str,
    seed: int,
    target_families: list[str] | None = None,
) -> None:
    """Refuse to reuse a filename whose saved selection settings differ."""
    if not report_path.is_file():
        raise SystemExit(
            f"existing BIOSCAN selection has no settings report: {report_path}; "
            "choose a new --dataset-tag instead of overwriting it"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid BIOSCAN selection report: {report_path}") from exc
    expected = {
        "requested": max_records,
        "max_per_taxon": max_per_taxon,
        "min_rank": min_rank,
        "seed": seed,
        "target_families": list(target_families or []),
    }
    mismatched = {
        key: {"saved": report.get(key), "requested": value}
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatched:
        raise SystemExit(
            "existing BIOSCAN selection settings differ; choose a new "
            f"--dataset-tag. Differences: {json.dumps(mismatched, sort_keys=True)}"
        )
    if int(report.get("selected", -1)) < 1:
        raise SystemExit(f"existing BIOSCAN selection report is incomplete: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/bioscan")
    parser.add_argument("--max-records", type=int, required=True)
    parser.add_argument(
        "--dataset-tag",
        help="Safe filename tag; defaults to a compact form of --max-records (for example 60k)",
    )
    parser.add_argument("--max-per-taxon", type=int, default=500)
    parser.add_argument("--min-rank", choices=("family", "genus", "species"), default="family")
    parser.add_argument(
        "--families-file",
        help="Optional versioned JSON scope; restrict the selection to these families",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--archive-config", default="configs/bioscan_archives_v06.json")
    parser.add_argument("--skip-metadata-download", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--no-suffix-range",
        action="store_true",
        help="Fallback for proxies that reject HTTP suffix ranges (uses HEAD + absolute ranges)",
    )
    args = parser.parse_args()
    if args.max_records < 1:
        raise SystemExit("--max-records must be positive")
    if args.max_per_taxon < 1:
        raise SystemExit("--max-per-taxon must be positive")

    tag = validate_tag(args.dataset_tag or count_tag(args.max_records))
    families_file = None
    target_families: list[str] = []
    if args.families_file:
        candidate = Path(args.families_file).expanduser()
        families_file = (
            candidate.resolve()
            if candidate.is_absolute()
            else (ROOT / candidate).resolve()
        )
        target_families, _ = load_family_filter(families_file)
    layout = dataset_paths(args.root, tag)
    store = layout["root"]
    store.mkdir(parents=True, exist_ok=True)

    if not args.skip_metadata_download:
        run(sys.executable, "scripts/download_bioscan.py", "--root", str(store), "--split", "all")
    metadata = find_metadata(store)
    print("BIOSCAN metadata:", metadata, flush=True)

    if not layout["selection"].exists():
        selection_command = [
            sys.executable,
            "scripts/select_bioscan_diptera.py",
            "--metadata", str(metadata),
            "--out", str(layout["selection"]),
            "--report", str(layout["selection_report"]),
            "--image-dir", str(layout["images"]),
            "--max-records", str(args.max_records),
            "--max-per-taxon", str(args.max_per_taxon),
            "--min-rank", args.min_rank,
            "--seed", str(args.seed),
        ]
        if families_file:
            selection_command.extend(["--families-file", str(families_file)])
        run(*selection_command)
    else:
        verify_existing_selection(
            layout["selection"],
            layout["selection_report"],
            max_records=args.max_records,
            max_per_taxon=args.max_per_taxon,
            min_rank=args.min_rank,
            seed=args.seed,
            target_families=target_families,
        )
        print(
            "Selection already exists; keeping it for reproducibility:",
            layout["selection"],
            flush=True,
        )

    if args.selection_only:
        print("Selection-only mode complete; no image bytes were downloaded.", flush=True)
        return

    command = [
        sys.executable,
        "scripts/download_bioscan_subset.py",
        "--selection", str(layout["selection"]),
        "--archive-config",
        str(
            (ROOT / args.archive_config).resolve()
            if not Path(args.archive_config).is_absolute()
            else Path(args.archive_config).resolve()
        ),
        "--image-dir", str(layout["images"]),
        "--out-manifest", str(layout["downloaded"]),
        "--report", str(layout["download_report"]),
        "--progress", str(layout["download_progress"]),
    ]
    if args.allow_partial:
        command.append("--allow-partial")
    if args.no_suffix_range:
        command.append("--no-suffix-range")
    run(*command)
    run(
        sys.executable,
        "scripts/ingest_bioscan.py",
        "--metadata", str(layout["downloaded"]),
        "--out", str(layout["manifest"]),
    )
    print(f"BIOSCAN {tag} READY: {layout['manifest']}", flush=True)


if __name__ == "__main__":
    main()
