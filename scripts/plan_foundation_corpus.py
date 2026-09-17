#!/usr/bin/env python3
"""Build a deterministic, source- and taxon-balanced research-scale training plan.

The input manifest can contain millions of rows.  Two streaming passes are used:
the first counts eligible strata and the second keeps only the deterministic
lowest-hash rows required by each source/taxon budget.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diptera_id.corpus.io import ManifestWriter, iter_table
from diptera_id.corpus.schema import as_bool, finalize_record


ROOT = Path(__file__).resolve().parents[1]


def sampling_bucket(record: dict, rank: str) -> str:
    if rank != "hierarchical":
        return str(record.get(rank, "")).strip()
    quality = str(record.get("label_quality", "")).strip().upper()
    if quality in {"A", "B"} and str(record.get("species", "")).strip():
        return f"species:{record['species']}"
    if str(record.get("genus", "")).strip():
        return f"genus:{record['genus']}"
    if str(record.get("family", "")).strip():
        return f"family:{record['family']}"
    return ""


def eligible(record: dict, rank: str, target_families: set[str] | None = None) -> bool:
    family = str(record.get("family", "")).strip().casefold()
    return bool(
        as_bool(record.get("eligible_supervised", ""))
        and sampling_bucket(record, rank)
        and (str(record.get("image_url", "")).strip() or str(record.get("local_path", "")).strip())
        and (not target_families or family in target_families)
    )


def load_target_families(profile: dict) -> tuple[list[str], set[str]]:
    """Load the versioned family scope used by every source in this plan."""
    names = [str(value).strip() for value in profile.get("target_families", []) if str(value).strip()]
    family_file = str(profile.get("target_families_file", "")).strip()
    if family_file:
        path = Path(family_file).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        names = [str(value).strip() for value in payload.get("families", []) if str(value).strip()]
    if len({name.casefold() for name in names}) != len(names):
        raise SystemExit("target family list contains duplicate names")
    return names, {name.casefold() for name in names}


def deterministic_score(seed: int, record_id: str) -> int:
    value = f"{seed}\x1f{record_id}".encode("utf-8")
    return int(hashlib.sha256(value).hexdigest()[:16], 16)


def allocate_source_budgets(available: Counter, requested: dict[str, int], total: int) -> dict[str, int]:
    """Allocate the requested mix, cap by availability, and redistribute capacity."""
    sources = [source for source in requested if available[source] > 0]
    budgets = {source: 0 for source in sources}
    remaining = min(total, sum(available[source] for source in sources))
    while remaining > 0:
        active = [source for source in sources if budgets[source] < available[source]]
        if not active:
            break
        progressed = 0
        weight_sum = sum(max(requested[source], 1) for source in active)
        for source in active:
            share = max(1, math.floor(remaining * max(requested[source], 1) / weight_sum))
            addition = min(share, available[source] - budgets[source], remaining - progressed)
            budgets[source] += addition
            progressed += addition
            if progressed >= remaining:
                break
        if progressed == 0:
            break
        remaining -= progressed
    return budgets


def allocate_taxon_budgets(capacities: dict[str, int], total: int, cap: int) -> dict[str, int]:
    """Square-root allocation: rare taxa are lifted without letting huge taxa dominate."""
    limits = {taxon: min(count, cap) for taxon, count in capacities.items() if count > 0}
    total = min(total, sum(limits.values()))
    budgets = {taxon: 0 for taxon in limits}
    remaining = total
    while remaining > 0:
        active = [taxon for taxon in limits if budgets[taxon] < limits[taxon]]
        if not active:
            break
        weights = {taxon: math.sqrt(limits[taxon] - budgets[taxon]) for taxon in active}
        weight_sum = sum(weights.values())
        progressed = 0
        for taxon in active:
            share = max(1, math.floor(remaining * weights[taxon] / weight_sum))
            addition = min(share, limits[taxon] - budgets[taxon], remaining - progressed)
            budgets[taxon] += addition
            progressed += addition
            if progressed >= remaining:
                break
        if progressed == 0:
            break
        remaining -= progressed
    return budgets


def load_profile(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Deduplicated master manifest")
    parser.add_argument("--config", default="configs/foundation_corpus_v05.json")
    parser.add_argument("--out", default="data/foundation_v05/training_plan.parquet")
    parser.add_argument("--report", default="data/foundation_v05/training_plan_report.json")
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()

    profile = load_profile(args.config)
    rank = str(profile.get("sampling_rank", "family"))
    if rank not in {"family", "genus", "species", "hierarchical"}:
        raise SystemExit(f"unsupported sampling rank: {rank}")
    total_target = int(profile.get("total_images", 100_000))
    source_targets = {str(key): int(value) for key, value in profile.get("source_targets", {}).items()}
    if not source_targets:
        raise SystemExit("config must define source_targets")
    per_taxon_cap = int(profile.get("max_per_source_taxon", 5_000))
    seed = int(profile.get("seed", 42))
    target_family_names, target_family_keys = load_target_families(profile)
    required_family_names = [str(value).strip() for value in profile.get("required_families", []) if str(value).strip()]

    source_counts: Counter = Counter()
    strata_counts: dict[str, Counter] = defaultdict(Counter)
    seen = 0
    for chunk in iter_table(args.input, args.chunksize):
        for raw in chunk.to_dict(orient="records"):
            record = finalize_record(raw)
            seen += 1
            if record["source"] not in source_targets or not eligible(record, rank, target_family_keys):
                continue
            source_counts[record["source"]] += 1
            strata_counts[record["source"]][sampling_bucket(record, rank)] += 1
        print(f"counted manifest rows: {seen}", end="\r")
    print()

    full_corpus = total_target == 0
    source_budgets = (
        {source: int(source_counts[source]) for source in source_targets if source_counts[source] > 0}
        if full_corpus
        else allocate_source_budgets(source_counts, source_targets, total_target)
    )
    stratum_budgets: dict[tuple[str, str], int] = {}
    for source, budget in source_budgets.items():
        cap = per_taxon_cap if per_taxon_cap > 0 else max(strata_counts[source].values(), default=0)
        for taxon, value in allocate_taxon_budgets(dict(strata_counts[source]), budget, cap).items():
            if value:
                stratum_budgets[(source, taxon)] = value

    heaps: dict[tuple[str, str], list[tuple[int, str, dict]]] = defaultdict(list)
    plan_out = Path(args.out)
    pending_out = plan_out.with_name(plan_out.stem + ".pending" + plan_out.suffix)
    direct_writer = ManifestWriter(pending_out) if full_corpus else None
    selected_direct = 0
    eligible_seen = 0
    selected_by_source: Counter = Counter()
    selected_by_taxon: Counter = Counter()
    selected_by_family: Counter = Counter()
    try:
        for chunk in iter_table(args.input, args.chunksize):
            direct_rows: list[dict] = []
            for raw in chunk.to_dict(orient="records"):
                record = finalize_record(raw)
                bucket = sampling_bucket(record, rank)
                key = (record["source"], bucket)
                budget = stratum_budgets.get(key, 0)
                if not budget or not eligible(record, rank, target_family_keys):
                    continue
                eligible_seen += 1
                if full_corpus:
                    direct_rows.append(record)
                    selected_direct += 1
                    selected_by_source[record["source"]] += 1
                    selected_by_taxon[f"{record['source']}\x1f{bucket}"] += 1
                    selected_by_family[record["family"]] += 1
                    continue
                score = deterministic_score(seed, record["record_id"])
                item = (-score, record["record_id"], record)
                heap = heaps[key]
                if len(heap) < budget:
                    heapq.heappush(heap, item)
                elif score < -heap[0][0]:
                    heapq.heapreplace(heap, item)
            if direct_writer is not None:
                direct_writer.write(direct_rows)
            print(f"selected through eligible rows: {eligible_seen}", end="\r")
    finally:
        if direct_writer is not None:
            direct_writer.close()
    print()

    selected: list[dict] = []
    if not full_corpus:
        for (source, taxon), heap in heaps.items():
            for _negative_score, _record_id, record in heap:
                selected.append(record)
                selected_by_source[source] += 1
                selected_by_taxon[f"{source}\x1f{taxon}"] += 1
                selected_by_family[record["family"]] += 1
        selected.sort(key=lambda row: (row["source"], sampling_bucket(row, rank), row["record_id"]))
        with ManifestWriter(pending_out) as writer:
            for start in range(0, len(selected), args.chunksize):
                writer.write(selected[start:start + args.chunksize])
    selected_total = selected_direct if full_corpus else len(selected)

    report = {
        "profile": profile.get("profile", "foundation-v0.5"),
        "input_rows_seen": seen,
        "eligible_rows_seen": eligible_seen,
        "requested_total": total_target,
        "selected_total": selected_total,
        "full_corpus": full_corpus,
        "sampling_rank": rank,
        "available_by_source": dict(source_counts),
        "budget_by_source": source_budgets,
        "selected_by_source": dict(selected_by_source),
        "target_families": target_family_names,
        "selected_by_family": dict(sorted(selected_by_family.items())),
        "taxa_selected": len(selected_by_taxon),
        "config": profile,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    missing = [source for source in source_targets if selected_by_source[source] == 0]
    selected_family_keys = {name.casefold() for name, count in selected_by_family.items() if count}
    missing_required_families = [name for name in required_family_names if name.casefold() not in selected_family_keys]
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if missing:
        raise SystemExit(f"training plan is missing requested sources: {', '.join(missing)}")
    if missing_required_families:
        raise SystemExit(
            "training plan is missing required target families: "
            + ", ".join(missing_required_families)
        )
    if selected_total == 0:
        raise SystemExit("Empty training plan; previous completed plan retained")
    pending_out.replace(plan_out)
    print(f"training plan -> {args.out}")


if __name__ == "__main__":
    main()
