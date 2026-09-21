#!/usr/bin/env python3
"""Select a deterministic, taxonomically balanced Diptera subset from BIOSCAN-5M.

Only the official metadata CSV is required.  The script makes two streaming
passes over the multi-million-row table, so the complete metadata never needs
to fit in RAM.  The output contains exact ZIP member names used by the
selective downloader; no image archive is downloaded during this step.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diptera_id.corpus.io import iter_table


VALID_SPLITS = (
    "pretrain",
    "train",
    "val",
    "test",
    "key_unseen",
    "val_unseen",
    "test_unseen",
    "other_heldout",
)

# A supervised foundation subset should not inherit BIOSCAN's 90% pretrain
# imbalance.  These weights retain broad family-level pretrain material while
# deliberately preserving the official seen/unseen evaluation partitions.
DEFAULT_SPLIT_TARGETS = {
    "pretrain": 12_000,
    "train": 12_000,
    "val": 1_500,
    "test": 1_500,
    "key_unseen": 750,
    "val_unseen": 750,
    "test_unseen": 750,
    "other_heldout": 750,
}

PLACEHOLDER = re.compile(r"(^|\s)(sp\.?|cf\.?|aff\.?)($|\s)|(^|_)sp\d+|^bold:", re.I)


def text(row: dict, *names: str) -> str:
    """Return the first non-empty metadata value, without turning NaN into text."""
    for name in names:
        value = row.get(name, "")
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            continue
        value = str(value).strip()
        if value and value.casefold() != "nan":
            return value
    return ""


def clean_species(value: str) -> str:
    return "" if not value or PLACEHOLDER.search(value) else value


def sampling_bucket(row: dict) -> tuple[str, str]:
    """Use the deepest trustworthy named rank for diversity balancing."""
    species = clean_species(text(row, "species", "species_name"))
    if species:
        return "species", species
    genus = text(row, "genus")
    if genus:
        return "genus", genus
    family = text(row, "family")
    if family:
        return "family", family
    return "", ""


def load_family_filter(value: str | Path | None) -> tuple[list[str], set[str]]:
    """Load an optional versioned family scope with case-insensitive matching."""
    if not value:
        return [], set()
    path = Path(value).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"target family file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid target family JSON {path}: {exc}") from exc
    values = payload.get("families", []) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise SystemExit(f"target family file must contain a families list: {path}")
    names = [str(value).strip() for value in values if str(value).strip()]
    keys = [name.casefold() for name in names]
    if not names:
        raise SystemExit(f"target family list is empty: {path}")
    if len(keys) != len(set(keys)):
        raise SystemExit(f"target family list contains duplicate names: {path}")
    return names, set(keys)


def is_eligible(
    row: dict,
    min_rank: str,
    target_families: set[str] | None = None,
) -> bool:
    if text(row, "order", "taxon_order").casefold() != "diptera":
        return False
    if not text(row, "processid", "process_id", "sampleid", "sample_id"):
        return False
    split = text(row, "split", "source_split")
    if split not in VALID_SPLITS:
        return False
    family = text(row, "family")
    genus = text(row, "genus")
    species = clean_species(text(row, "species", "species_name"))
    requirements = {
        "family": bool(family),
        "genus": bool(genus),
        "species": bool(species),
    }
    return requirements[min_rank] and (
        not target_families or family.casefold() in target_families
    )


def stable_score(seed: int, process_id: str) -> int:
    payload = f"{seed}\x1f{process_id}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def allocate_budgets(available: Counter, requested: dict[str, int], total: int) -> dict[str, int]:
    """Respect requested weights, cap unavailable strata, and redistribute."""
    keys = [key for key in requested if available[key] > 0]
    budgets = {key: 0 for key in keys}
    remaining = min(total, sum(available[key] for key in keys))
    while remaining:
        active = [key for key in keys if budgets[key] < available[key]]
        if not active:
            break
        weight_sum = sum(max(requested[key], 1) for key in active)
        progressed = 0
        for key in active:
            share = max(1, math.floor(remaining * max(requested[key], 1) / weight_sum))
            addition = min(share, available[key] - budgets[key], remaining - progressed)
            budgets[key] += addition
            progressed += addition
            if progressed >= remaining:
                break
        if not progressed:
            break
        remaining -= progressed
    return budgets


def allocate_taxon_budgets(capacities: Counter, total: int, cap: int) -> dict[str, int]:
    """Square-root allocation lifts rare taxa and prevents common-family dominance."""
    limits = {key: min(count, cap) for key, count in capacities.items() if count > 0}
    total = min(total, sum(limits.values()))
    budgets = {key: 0 for key in limits}
    remaining = total
    while remaining:
        active = [key for key in limits if budgets[key] < limits[key]]
        if not active:
            break
        weights = {key: math.sqrt(limits[key] - budgets[key]) for key in active}
        weight_sum = sum(weights.values())
        progressed = 0
        for key in active:
            share = max(1, math.floor(remaining * weights[key] / weight_sum))
            addition = min(share, limits[key] - budgets[key], remaining - progressed)
            budgets[key] += addition
            progressed += addition
            if progressed >= remaining:
                break
        if not progressed:
            break
        remaining -= progressed
    return budgets


def parse_split_targets(value: str | None) -> dict[str, int]:
    if not value:
        return dict(DEFAULT_SPLIT_TARGETS)
    path = Path(value)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {}
        for item in value.split(","):
            name, amount = item.split("=", 1)
            payload[name.strip()] = int(amount)
    unknown = set(payload) - set(VALID_SPLITS)
    if unknown:
        raise SystemExit(f"unknown BIOSCAN split(s): {', '.join(sorted(unknown))}")
    return {split: int(payload.get(split, 0)) for split in VALID_SPLITS}


def archive_group(split: str) -> str:
    if split == "pretrain":
        return "pretrain"
    if split == "train":
        return "train"
    return "eval"


def archive_member(row: dict) -> str:
    split = text(row, "split", "source_split")
    chunk = text(row, "chunk")
    process_id = text(row, "processid", "process_id", "sampleid", "sample_id")
    middle = f"{chunk}/" if chunk else ""
    return f"{split}/{middle}{process_id}.jpg"


def prepare_selected_row(row: dict, image_dir: Path) -> dict:
    result = {str(key): "" if pd.isna(value) else value for key, value in row.items()}
    process_id = text(result, "processid", "process_id", "sampleid", "sample_id")
    split = text(result, "split", "source_split")
    rank, bucket = sampling_bucket(result)
    result.update({
        "processid": process_id,
        "order": "Diptera",
        "species": clean_species(text(result, "species", "species_name")),
        "source_split": split,
        "selection_rank": rank,
        "selection_bucket": bucket,
        "archive_group": archive_group(split),
        "archive_member": archive_member(result),
        "local_path": str((image_dir / f"{process_id}.jpg").resolve()),
        "image_license": "CC BY 3.0",
        "copyright_holder": "CBG Photography Group",
        "attribution": "CBG Robotic Imager",
        "publisher": "Centre for Biodiversity Genomics",
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="Official BIOSCAN-5M metadata CSV")
    parser.add_argument("--out", default="data/raw/bioscan/diptera_30k_selection.csv")
    parser.add_argument("--report", default="data/raw/bioscan/diptera_30k_selection_report.json")
    parser.add_argument("--image-dir", default="data/raw/bioscan/diptera_30k_images")
    parser.add_argument("--max-records", type=int, default=30_000)
    parser.add_argument("--max-per-taxon", type=int, default=500)
    parser.add_argument("--min-rank", choices=("family", "genus", "species"), default="family")
    parser.add_argument(
        "--families-file",
        help="Optional JSON family scope; only matching Diptera are eligible",
    )
    parser.add_argument("--split-targets", help="JSON path or comma list such as pretrain=12000,train=12000,...")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--allow-shortfall", action="store_true")
    args = parser.parse_args()
    if args.max_records <= 0:
        raise SystemExit("--max-records must be positive")
    if args.max_per_taxon <= 0:
        raise SystemExit("--max-per-taxon must be positive")

    target_family_names, target_family_keys = load_family_filter(args.families_file)

    split_targets = parse_split_targets(args.split_targets)
    split_counts: Counter = Counter()
    bucket_counts: dict[str, Counter] = defaultdict(Counter)
    eligible_total = 0

    # Pass 1: count only eligible Diptera strata.
    for chunk in iter_table(args.metadata, args.chunksize):
        for row in chunk.to_dict(orient="records"):
            if not is_eligible(row, args.min_rank, target_family_keys):
                continue
            split = text(row, "split", "source_split")
            rank, bucket = sampling_bucket(row)
            if not bucket:
                continue
            key = f"{rank}:{bucket}"
            split_counts[split] += 1
            bucket_counts[split][key] += 1
            eligible_total += 1
        print(f"BIOSCAN metadata pass 1: {eligible_total:,} eligible Diptera", end="\r")
    print()

    split_budgets = allocate_budgets(split_counts, split_targets, args.max_records)
    bucket_budgets: dict[tuple[str, str], int] = {}
    for split, budget in split_budgets.items():
        allocation = allocate_taxon_budgets(bucket_counts[split], budget, args.max_per_taxon)
        for bucket, amount in allocation.items():
            if amount:
                bucket_budgets[(split, bucket)] = amount

    # Pass 2: keep only the lowest deterministic hashes required by each budget.
    heaps: dict[tuple[str, str], list[tuple[int, str, dict]]] = defaultdict(list)
    considered = 0
    for chunk in iter_table(args.metadata, args.chunksize):
        for row in chunk.to_dict(orient="records"):
            if not is_eligible(row, args.min_rank, target_family_keys):
                continue
            split = text(row, "split", "source_split")
            rank, bucket_value = sampling_bucket(row)
            key = (split, f"{rank}:{bucket_value}")
            budget = bucket_budgets.get(key, 0)
            if not budget:
                continue
            process_id = text(row, "processid", "process_id", "sampleid", "sample_id")
            score = stable_score(args.seed, process_id)
            item = (-score, process_id, row)
            heap = heaps[key]
            if len(heap) < budget:
                heapq.heappush(heap, item)
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, item)
            considered += 1
        print(f"BIOSCAN metadata pass 2: {considered:,} candidates", end="\r")
    print()

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    selected = [prepare_selected_row(row, image_dir) for heap in heaps.values() for _, _, row in heap]
    selected.sort(key=lambda row: (row["source_split"], row["selection_rank"], row["selection_bucket"], row["processid"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected).to_csv(out, index=False)

    selected_splits = Counter(row["source_split"] for row in selected)
    selected_ranks = Counter(row["selection_rank"] for row in selected)
    selected_families = {text(row, "family") for row in selected if text(row, "family")}
    selected_genera = {text(row, "genus") for row in selected if text(row, "genus")}
    selected_species = {clean_species(text(row, "species")) for row in selected if clean_species(text(row, "species"))}
    report = {
        "requested": args.max_records,
        "selected": len(selected),
        "eligible_diptera": eligible_total,
        "min_rank": args.min_rank,
        "seed": args.seed,
        "max_per_taxon": args.max_per_taxon,
        "target_families": target_family_names,
        "available_by_split": dict(split_counts),
        "budget_by_split": split_budgets,
        "selected_by_split": dict(selected_splits),
        "selected_by_deepest_rank": dict(selected_ranks),
        "taxonomic_coverage": {
            "families": len(selected_families),
            "genera": len(selected_genera),
            "species": len(selected_species),
        },
        "full_image_archives_downloaded": False,
        "next_step": "download_bioscan_subset.py",
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"BIOSCAN selection -> {out}")
    if len(selected) < args.max_records and not args.allow_shortfall:
        raise SystemExit(
            f"selected only {len(selected):,}/{args.max_records:,}; use --allow-shortfall or relax --min-rank/--max-per-taxon"
        )


if __name__ == "__main__":
    main()
