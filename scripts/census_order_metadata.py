#!/usr/bin/env python3
"""Audit taxonomic coverage for one insect order without downloading images."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ingest_inat import load_taxa


def clean_values(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def add_unique(target: set[str], values: pd.Series) -> None:
    target.update(value for value in clean_values(values) if value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", default="Hymenoptera")
    parser.add_argument("--inat-taxa", required=True)
    parser.add_argument("--bioscan-metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    order_name = args.order.strip()
    if not order_name:
        raise SystemExit("--order must not be empty")
    if args.chunksize < 1:
        raise SystemExit("--chunksize must be positive")

    print(f"Loading iNaturalist taxonomy for {order_name}…", flush=True)
    taxa, selected_taxa = load_taxa(
        args.inat_taxa,
        args.chunksize,
        order_name,
    )

    inat_by_rank: Counter[str] = Counter()
    inat_names_by_rank: dict[str, set[str]] = defaultdict(set)

    for taxon_id in selected_taxa:
        _ancestry, rank, name = taxa[taxon_id]
        rank = rank or "unranked"
        inat_by_rank[rank] += 1
        if name:
            inat_names_by_rank[rank].add(name)

    print(
        f"iNaturalist taxonomy: {len(selected_taxa):,} taxa; "
        f"{len(inat_names_by_rank['family']):,} families",
        flush=True,
    )

    metadata = Path(args.bioscan_metadata)
    header = pd.read_csv(metadata, nrows=0).columns.tolist()

    def column(*candidates: str) -> str | None:
        mapping = {value.casefold(): value for value in header}
        for candidate in candidates:
            if candidate.casefold() in mapping:
                return mapping[candidate.casefold()]
        return None

    order_column = column("order", "taxon_order")
    if not order_column:
        raise SystemExit("BIOSCAN metadata does not contain an order column")

    columns = {
        "order": order_column,
        "processid": column("processid", "process_id", "sampleid", "sample_id"),
        "family": column("family"),
        "genus": column("genus"),
        "species": column("species", "species_name"),
        "dna_bin": column("dna_bin", "bin_uri"),
        "split": column("split", "source_split"),
    }

    usecols = list(dict.fromkeys(
        value for value in columns.values() if value is not None
    ))

    total_rows = 0
    selected_rows = 0
    rows_with_process_id = 0
    rows_with_family = 0
    rows_with_genus = 0
    rows_with_species = 0
    rows_with_dna_bin = 0

    split_counts: Counter[str] = Counter()
    unique_families: set[str] = set()
    unique_genera: set[str] = set()
    unique_species: set[str] = set()
    unique_dna_bins: set[str] = set()

    family_stats: dict[str, dict] = defaultdict(
        lambda: {
            "rows": 0,
            "with_genus": 0,
            "with_species": 0,
            "with_dna_bin": 0,
            "genera": set(),
            "species": set(),
            "dna_bins": set(),
        }
    )

    print(f"Scanning BIOSCAN metadata for {order_name}…", flush=True)

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            metadata,
            usecols=usecols,
            dtype=str,
            keep_default_na=False,
            chunksize=args.chunksize,
            low_memory=False,
        ),
        start=1,
    ):
        total_rows += len(chunk)

        order_values = clean_values(chunk[columns["order"]])
        selected = chunk[
            order_values.str.casefold() == order_name.casefold()
        ].copy()

        if selected.empty:
            print(
                f"BIOSCAN chunk {chunk_number}: "
                f"{total_rows:,} total / {selected_rows:,} selected",
                flush=True,
            )
            continue

        selected_rows += len(selected)

        normalized: dict[str, pd.Series] = {}
        for logical_name in (
            "processid", "family", "genus", "species", "dna_bin", "split"
        ):
            source_column = columns[logical_name]
            if source_column:
                normalized[logical_name] = clean_values(selected[source_column])
            else:
                normalized[logical_name] = pd.Series(
                    [""] * len(selected),
                    index=selected.index,
                    dtype=str,
                )

        rows_with_process_id += int(normalized["processid"].ne("").sum())
        rows_with_family += int(normalized["family"].ne("").sum())
        rows_with_genus += int(normalized["genus"].ne("").sum())
        rows_with_species += int(normalized["species"].ne("").sum())
        rows_with_dna_bin += int(normalized["dna_bin"].ne("").sum())

        split_counts.update(
            value or "__missing__" for value in normalized["split"]
        )

        add_unique(unique_families, normalized["family"])
        add_unique(unique_genera, normalized["genus"])
        add_unique(unique_species, normalized["species"])
        add_unique(unique_dna_bins, normalized["dna_bin"])

        selected["__family"] = normalized["family"].replace(
            "", "__unresolved__"
        )
        selected["__genus"] = normalized["genus"]
        selected["__species"] = normalized["species"]
        selected["__dna_bin"] = normalized["dna_bin"]

        for family, group in selected.groupby("__family", sort=False):
            stats = family_stats[str(family)]
            stats["rows"] += len(group)
            stats["with_genus"] += int(group["__genus"].ne("").sum())
            stats["with_species"] += int(group["__species"].ne("").sum())
            stats["with_dna_bin"] += int(group["__dna_bin"].ne("").sum())
            add_unique(stats["genera"], group["__genus"])
            add_unique(stats["species"], group["__species"])
            add_unique(stats["dna_bins"], group["__dna_bin"])

        print(
            f"BIOSCAN chunk {chunk_number}: "
            f"{total_rows:,} total / {selected_rows:,} {order_name}",
            flush=True,
        )

    family_report = []
    for family, stats in family_stats.items():
        family_report.append({
            "family": family,
            "rows": stats["rows"],
            "with_genus": stats["with_genus"],
            "with_species": stats["with_species"],
            "with_dna_bin": stats["with_dna_bin"],
            "unique_genera": len(stats["genera"]),
            "unique_species": len(stats["species"]),
            "unique_dna_bins": len(stats["dna_bins"]),
        })

    family_report.sort(key=lambda row: (-row["rows"], row["family"]))

    report = {
        "schema": 1,
        "taxonomic_order": order_name,
        "inaturalist_taxonomy": {
            "all_taxa_in_table": len(taxa),
            "order_taxa": len(selected_taxa),
            "by_rank": dict(sorted(inat_by_rank.items())),
            "unique_names_by_rank": {
                rank: len(names)
                for rank, names in sorted(inat_names_by_rank.items())
            },
            "families": sorted(inat_names_by_rank["family"]),
        },
        "bioscan_metadata": {
            "all_rows_scanned": total_rows,
            "order_rows": selected_rows,
            "rows_with_process_id": rows_with_process_id,
            "rows_with_family": rows_with_family,
            "rows_with_genus": rows_with_genus,
            "rows_with_species": rows_with_species,
            "rows_with_dna_bin": rows_with_dna_bin,
            "unique_families": len(unique_families),
            "unique_genera": len(unique_genera),
            "unique_species": len(unique_species),
            "unique_dna_bins": len(unique_dna_bins),
            "by_split": dict(sorted(split_counts.items())),
            "families": family_report,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pending = out.with_suffix(out.suffix + ".pending")
    pending.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pending.replace(out)

    print(json.dumps({
        "order": order_name,
        "inat_order_taxa": len(selected_taxa),
        "inat_families": len(inat_names_by_rank["family"]),
        "bioscan_rows": selected_rows,
        "bioscan_families": len(unique_families),
        "bioscan_genera": len(unique_genera),
        "bioscan_species": len(unique_species),
    }, indent=2))
    print(f"Census report -> {out}")


if __name__ == "__main__":
    main()
