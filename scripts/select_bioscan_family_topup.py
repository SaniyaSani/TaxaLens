#!/usr/bin/env python3
"""Select only the BIOSCAN rows needed to top up named target families.

Existing valid images are counted first. Missing rows already present in the
original selection are preferred, then additional deterministic rows are drawn
from the local BIOSCAN metadata. This script never downloads image bytes.
"""
from __future__ import annotations

import argparse
import heapq
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from download_bioscan_subset import output_path, valid_existing
from select_bioscan_diptera import (
    allocate_taxon_budgets,
    is_eligible,
    prepare_selected_row,
    sampling_bucket,
    stable_score,
    text,
)
from diptera_id.corpus.io import iter_table


def family_name(row: dict) -> str:
    return text(row, "family")


def process_id(row: dict) -> str:
    return text(row, "processid", "process_id", "sampleid", "sample_id", "specimen_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--base-selection", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--families", default="Muscidae,Tachinidae")
    parser.add_argument("--min-per-family", type=int, default=1500)
    parser.add_argument("--max-per-taxon", type=int, default=250)
    parser.add_argument("--min-rank", choices=("family", "genus", "species"), default="family")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--allow-shortfall", action="store_true")
    args = parser.parse_args()
    if args.min_per_family <= 0 or args.max_per_taxon <= 0:
        raise SystemExit("--min-per-family and --max-per-taxon must be positive")

    requested_names = [value.strip() for value in args.families.split(",") if value.strip()]
    requested = {name.casefold(): name for name in requested_names}
    if not requested:
        raise SystemExit("no target families supplied")

    image_dir = Path(args.image_dir).expanduser().resolve()
    base = pd.read_csv(args.base_selection, dtype=str, keep_default_na=False)
    base_rows = base.to_dict(orient="records")
    base_ids = {process_id(row) for row in base_rows if process_id(row)}
    selected: list[dict] = []
    deficits: dict[str, int] = {}
    report_by_family: dict[str, dict] = {}

    for family_key, display_name in requested.items():
        rows = [row for row in base_rows if family_name(row).casefold() == family_key]
        valid_rows = [row for row in rows if valid_existing(output_path(row, image_dir))]
        missing_rows = [row for row in rows if not valid_existing(output_path(row, image_dir))]
        valid_rows.sort(key=lambda row: stable_score(args.seed, process_id(row)))
        missing_rows.sort(key=lambda row: stable_score(args.seed, process_id(row)))
        chosen = valid_rows[: args.min_per_family]
        remaining = max(0, args.min_per_family - len(chosen))
        chosen_missing = missing_rows[:remaining]
        chosen.extend(chosen_missing)
        remaining -= len(chosen_missing)
        selected.extend(chosen)
        deficits[family_key] = remaining
        report_by_family[display_name] = {
            "target": args.min_per_family,
            "existing_valid": len(valid_rows),
            "available_in_base_selection": len(rows),
            "chosen_missing_from_base_selection": len(chosen_missing),
            "new_metadata_rows": 0,
        }

    # Count diversity strata only for the still-missing family quotas.
    bucket_counts: dict[str, Counter] = defaultdict(Counter)
    for chunk_number, chunk in enumerate(iter_table(args.metadata, args.chunksize), 1):
        print(f"BIOSCAN metadata: counting chunk {chunk_number} ({len(chunk)} rows)", flush=True)
        for row in chunk.to_dict(orient="records"):
            family_key = family_name(row).casefold()
            if deficits.get(family_key, 0) <= 0 or process_id(row) in base_ids:
                continue
            if not is_eligible(row, args.min_rank):
                continue
            split = text(row, "split", "source_split")
            rank, bucket = sampling_bucket(row)
            if bucket:
                bucket_counts[family_key][f"{split}|{rank}:{bucket}"] += 1

    budgets: dict[tuple[str, str], int] = {}
    for family_key, deficit in deficits.items():
        if deficit <= 0:
            continue
        for bucket, amount in allocate_taxon_budgets(
            bucket_counts[family_key], deficit, args.max_per_taxon
        ).items():
            if amount:
                budgets[(family_key, bucket)] = amount

    heaps: dict[tuple[str, str], list[tuple[int, str, dict]]] = defaultdict(list)
    if budgets:
        for chunk_number, chunk in enumerate(iter_table(args.metadata, args.chunksize), 1):
            print(f"BIOSCAN metadata: selecting chunk {chunk_number} ({len(chunk)} rows)", flush=True)
            for row in chunk.to_dict(orient="records"):
                family_key = family_name(row).casefold()
                identifier = process_id(row)
                if deficits.get(family_key, 0) <= 0 or identifier in base_ids:
                    continue
                if not is_eligible(row, args.min_rank):
                    continue
                split = text(row, "split", "source_split")
                rank, bucket_value = sampling_bucket(row)
                key = (family_key, f"{split}|{rank}:{bucket_value}")
                budget = budgets.get(key, 0)
                if not budget:
                    continue
                score = stable_score(args.seed, identifier)
                item = (-score, identifier, row)
                heap = heaps[key]
                if len(heap) < budget:
                    heapq.heappush(heap, item)
                elif score < -heap[0][0]:
                    heapq.heapreplace(heap, item)

    new_by_family: Counter = Counter()
    for (family_key, _bucket), heap in heaps.items():
        for _negative_score, _identifier, row in heap:
            selected.append(prepare_selected_row(row, image_dir))
            new_by_family[family_key] += 1
    selected.sort(key=lambda row: (family_name(row).casefold(), text(row, "source_split", "split"), process_id(row)))

    final_counts = Counter(family_name(row).casefold() for row in selected)
    for family_key, display_name in requested.items():
        report_by_family[display_name]["new_metadata_rows"] = new_by_family[family_key]
        report_by_family[display_name]["selected_total"] = final_counts[family_key]
        report_by_family[display_name]["shortfall"] = max(0, args.min_per_family - final_counts[family_key])

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected).to_csv(out, index=False)
    report = {
        "mode": "target_family_topup",
        "families": requested_names,
        "min_per_family": args.min_per_family,
        "base_selection_rows": len(base_rows),
        "topup_selection_rows": len(selected),
        "by_family": report_by_family,
        "network_downloads_during_selection": 0,
    }
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"BIOSCAN target-family top-up selection -> {out}")
    shortfalls = [name for name, values in report_by_family.items() if values["shortfall"]]
    if shortfalls and not args.allow_shortfall:
        raise SystemExit("BIOSCAN metadata cannot meet target for: " + ", ".join(shortfalls))


if __name__ == "__main__":
    main()
