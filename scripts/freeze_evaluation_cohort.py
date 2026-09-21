#!/usr/bin/env python3
"""Freeze the quality-approved validation/test cohort for scale comparisons."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from recovery_support import atomic_json, digest, nonempty

from diptera_id.corpus.io import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    destination = Path(args.out).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    if not nonempty(source):
        raise SystemExit(f"quality-approved manifest missing or empty: {source}")
    source_sha = digest(source)

    if nonempty(destination) and nonempty(report_path):
        try:
            saved = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            saved = {}
        if (
            saved.get("input_sha256") == source_sha
            and saved.get("cohort_sha256") == digest(destination)
        ):
            print(f"Reusing frozen evaluation cohort: {destination}")
            return
        raise SystemExit(
            "STOP: a different frozen evaluation cohort already exists. "
            "Choose a new output path; the existing cohort was not overwritten."
        )

    frame = load_manifest(source).fillna("")
    required = {"split", "source", "family"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"manifest is missing evaluation columns: {sorted(missing)}")
    unknown = set(frame["split"].astype(str)) - {"train", "val", "test"}
    if unknown:
        raise SystemExit(f"unknown split values: {sorted(unknown)}")
    group_column = next(
        (name for name in ("split_group", "duplicate_group_id", "specimen_group_id", "record_id") if name in frame),
        None,
    )
    if not group_column:
        raise SystemExit("manifest has no group identifier for leakage validation")
    groups = frame[group_column].astype(str)
    if groups.eq("").any() and "record_id" in frame:
        groups = groups.where(groups.ne(""), "record:" + frame["record_id"].astype(str))
    if (pd.DataFrame({"group": groups, "split": frame["split"]}).groupby("group")["split"].nunique() > 1).any():
        raise SystemExit("split leakage: one specimen/duplicate group occurs in multiple splits")

    cohort = frame[frame["split"].isin(["val", "test"])].copy()
    if cohort.empty:
        raise SystemExit("no validation/test rows to freeze")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.stem + ".pending" + destination.suffix)
    cohort.to_parquet(pending, index=False)
    pending.replace(destination)
    report = {
        "schema": 1,
        "input": str(source),
        "input_sha256": source_sha,
        "cohort": str(destination),
        "cohort_sha256": digest(destination),
        "rows": len(cohort),
        "group_column": group_column,
        "split_counts": {str(key): int(value) for key, value in cohort["split"].value_counts().items()},
        "source_counts": {str(key): int(value) for key, value in cohort["source"].value_counts().items()},
        "family_counts": {str(key): int(value) for key, value in cohort["family"].value_counts().items()},
        "rule": "quality-approved val/test rows only; specimen/duplicate groups cannot cross splits",
    }
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
