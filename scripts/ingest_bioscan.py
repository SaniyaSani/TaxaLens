#!/usr/bin/env python3
"""Normalize BIOSCAN-5M image and specimen metadata for a selected order."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diptera_id.corpus.io import ManifestWriter, iter_table
from diptera_id.corpus.schema import finalize_record, first


PLACEHOLDER = re.compile(r"(^|\s)(sp\.?|cf\.?|aff\.?)($|\s)|(^|_)sp\d+|^bold:", re.I)


def clean_species(value: str) -> str:
    return "" if PLACEHOLDER.search(value or "") else value


def to_record(
    row: dict,
    image_root: Path | None,
    image_extension: str,
    assume_diptera: bool,
    order_name: str = "Diptera",
) -> dict | None:
    order = first(row, ["order", "taxon_order"])
    if not assume_diptera and order.casefold() != order_name.casefold():
        return None
    process_id = first(row, ["processid", "process_id", "sampleid", "sample_id", "specimen_id"])
    if not process_id:
        return None
    species_raw = first(row, ["species", "species_name"])
    species = clean_species(species_raw)
    local_path = first(row, ["local_path", "image_path", "image_file", "file_name", "filename"])
    if image_root:
        candidate = Path(local_path) if local_path else Path(f"{process_id}{image_extension}")
        local_path = str((image_root / candidate).resolve())
    image_url = first(row, ["image_url", "image_uri", "url", "identifier"])
    dna = first(row, ["dna_barcode", "nuc", "sequence", "barcode_sequence"])
    record = finalize_record({
        "source": "BIOSCAN-5M",
        "source_record_id": process_id,
        "source_image_id": process_id,
        "image_url": image_url,
        "local_path": local_path,
        "image_license": first(row, ["license", "image_license"], "CC BY 3.0"),
        "copyright_holder": first(row, ["copyright_holder"], "CBG Photography Group"),
        "attribution": first(row, ["attribution", "photographer"], "CBG Robotic Imager"),
        "publisher": first(row, ["publisher"], "Centre for Biodiversity Genomics"),
        "source_dataset": "BIOSCAN-5M",
        "source_url": first(row, ["record_url", "source_url"]),
        "basis_of_record": "PRESERVED_SPECIMEN",
        "is_preserved_specimen": True,
        "kingdom": first(row, ["kingdom"]),
        "phylum": first(row, ["phylum"]),
        "class": first(row, ["class"]),
        "order": order or order_name,
        "family": first(row, ["family"]),
        "subfamily": first(row, ["subfamily"]),
        "genus": first(row, ["genus"]),
        "species": species,
        "original_scientific_name": species_raw or first(row, ["taxon"]),
        "identified_by": first(row, ["identified_by", "identifier"]),
        "country": first(row, ["country"]),
        "state_province": first(row, ["province_state", "province/state", "state_province"]),
        "latitude": first(row, ["latitude", "coord-lat", "coord_lat"]),
        "longitude": first(row, ["longitude", "coord-lon", "coord_lon"]),
        "event_date": first(row, ["event_date", "collection_date"]),
        "sex": first(row, ["sex"]),
        "life_stage": first(row, ["life_stage", "stage"]),
        "dna_barcode": dna,
        "dna_bin": first(row, ["dna_bin", "bin", "barcode_index_number"]),
        "specimen_group_id": f"BIOSCAN-5M:{process_id}",
        "source_split": first(row, ["source_split", "split", "partition"]),
        "label_quality": "A" if dna and species else ("B" if species else "D"),
    })
    if not record["image_url"] and not record["local_path"]:
        record["eligible_supervised"] = False
        record["exclusion_reason"] = "missing_image_location"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="Official BIOSCAN metadata CSV/JSONL/Parquet")
    parser.add_argument("--image-root", help="Root containing extracted BIOSCAN image files")
    parser.add_argument("--image-extension", default=".jpg")
    parser.add_argument("--out", default="data/corpus/bioscan_raw.parquet")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument(
        "--order",
        default="Diptera",
        help="Taxonomic order to retain",
    )
    parser.add_argument(
        "--assume-diptera",
        action="store_true",
        help="Legacy option for already filtered Diptera input",
    )
    args = parser.parse_args()

    order_name = args.order.strip()
    if not order_name:
        raise SystemExit("--order must not be empty")
    if args.assume_diptera and order_name.casefold() != "diptera":
        raise SystemExit(
            "--assume-diptera cannot be combined with a non-Diptera --order"
        )

    root = Path(args.image_root) if args.image_root else None
    with ManifestWriter(args.out) as writer:
        for chunk in iter_table(args.metadata, args.chunksize):
            rows = [
                record
                for raw in chunk.to_dict(orient="records")
                if (
                    record := to_record(
                        raw,
                        root,
                        args.image_extension,
                        args.assume_diptera,
                        order_name,
                    )
                )
            ]
            writer.write(rows)
            print(
                f"wrote BIOSCAN {order_name}: {writer.rows_written}",
                end="\r",
            )
        print(
            f"\nwrote {writer.rows_written} BIOSCAN-5M "
            f"{order_name} rows -> {args.out}"
        )


if __name__ == "__main__":
    main()
