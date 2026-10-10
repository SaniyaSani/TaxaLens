#!/usr/bin/env python3
"""Normalize a GBIF occurrence download plus multimedia table.

The expected input is an extracted DWCA download (occurrence.txt and
multimedia.txt). A disk-backed join keeps RAM use bounded on large downloads.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diptera_id.corpus.io import ManifestWriter, iter_table
from diptera_id.corpus.join import RecordIndex, rows_from_chunk
from diptera_id.corpus.schema import finalize_record, first, normalize_license


def occurrence_record(
    row: dict,
    assume_filtered: bool,
    order_name: str = "Diptera",
) -> tuple[str, dict] | None:
    order = first(row, ["order"])
    basis = first(row, ["basisofrecord", "basis_of_record"]).upper()
    if not assume_filtered and (
        order.casefold() != order_name.casefold()
        or basis != "PRESERVED_SPECIMEN"
    ):
        return None
    join_id = first(row, ["gbifid", "id", "occurrenceid", "coreid"])
    if not join_id:
        return None
    species = first(row, ["species", "acceptedscientificname"])
    record = {
        "source": "GBIF",
        "source_record_id": join_id,
        "order": order or order_name,
        "family": first(row, ["family"]),
        "subfamily": first(row, ["subfamily"]),
        "tribe": first(row, ["tribe"]),
        "genus": first(row, ["genus"]),
        "species": species,
        "subspecies": first(row, ["infraspecificepithet", "subspecies"]),
        "original_scientific_name": first(row, ["scientificname", "verbatimscientificname"]),
        "accepted_scientific_name": first(row, ["acceptedscientificname"]),
        "taxon_id": first(row, ["taxonkey", "specieskey", "taxonid"]),
        "identified_by": first(row, ["identifiedby"]),
        "identification_remarks": first(row, ["identificationremarks", "taxonremarks"]),
        "country": first(row, ["country", "countrycode"]),
        "state_province": first(row, ["stateprovince"]),
        "latitude": first(row, ["decimallatitude"]),
        "longitude": first(row, ["decimallongitude"]),
        "elevation_m": first(row, ["elevation", "minimumelevationinmeters"]),
        "event_date": first(row, ["eventdate", "dateidentified"]),
        "sex": first(row, ["sex"]),
        "life_stage": first(row, ["lifestage"]),
        "publisher": first(row, ["publishingorgkey", "institutioncode", "ownerinstitutioncode"]),
        "source_dataset": first(row, ["datasettitle", "datasetname", "datasetkey"]),
        "source_url": first(row, ["references", "occurrenceid"]),
        "basis_of_record": basis or "PRESERVED_SPECIMEN",
        "is_preserved_specimen": True,
        "specimen_group_id": f"GBIF:{join_id}",
        "label_quality": "B" if species else "D",
        # Some SIMPLE_CSV exports contain one media URL directly.
        "image_url": first(row, ["media", "identifier", "image_url"]),
        "image_license": first(row, ["license", "media_license"]),
    }
    return join_id, record


def multimedia_record(row: dict, base: dict) -> dict | None:
    media_type = first(row, ["type", "mediatype", "format"]).lower()
    if media_type and not any(value in media_type for value in ("stillimage", "image", "jpeg", "jpg", "png", "webp")):
        return None
    url = first(row, ["identifier", "accessuri", "references", "image_url", "url"])
    if not url.startswith(("http://", "https://")):
        return None
    record = dict(base)
    record.update({
        "source_image_id": first(row, ["id", "identifier", "mediaid"], url),
        "image_url": url,
        "image_license": first(row, ["license", "licence", "rights", "rightsuri"]),
        "copyright_holder": first(row, ["rightsholder", "creator", "owner"]),
        "attribution": first(row, ["creator", "title", "description"]),
    })
    return finalize_record(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occurrence", required=True, help="Extracted GBIF occurrence.txt/CSV/Parquet")
    parser.add_argument("--multimedia", help="Extracted multimedia.txt. Omit only for SIMPLE_CSV with media column.")
    parser.add_argument("--out", default="data/corpus/gbif_raw.parquet")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--licenses", default="CC0,CC-BY,CC-BY-SA")
    parser.add_argument("--keep-ineligible", action="store_true")
    parser.add_argument(
        "--order",
        default="Diptera",
        help="Taxonomic order to retain",
    )
    parser.add_argument(
        "--assume-filtered",
        action="store_true",
        help="Input already contains only preserved specimens from --order",
    )
    args = parser.parse_args()

    order_name = args.order.strip()
    if not order_name:
        raise SystemExit("--order must not be empty")

    allowed = {normalize_license(value) for value in args.licenses.split(",")}
    with RecordIndex() as index:
        indexed = 0
        for chunk in iter_table(args.occurrence, args.chunksize):
            indexed += index.add(
                rows_from_chunk(
                    chunk,
                    lambda row: occurrence_record(
                        row,
                        args.assume_filtered,
                        order_name,
                    ),
                )
            )
            print(
                f"indexed GBIF preserved {order_name}: {indexed}",
                end="\r",
            )
        print(f"\nindexed GBIF preserved {order_name}: {indexed}")

        with ManifestWriter(args.out) as writer:
            if args.multimedia:
                for chunk in iter_table(args.multimedia, args.chunksize):
                    raw_rows = chunk.to_dict(orient="records")
                    join_ids = [first(row, ["coreid", "gbifid", "occurrenceid", "parentid"]) for row in raw_rows]
                    bases = index.get_many(join_ids)
                    output = []
                    for row, join_id in zip(raw_rows, join_ids):
                        if join_id not in bases:
                            continue
                        record = multimedia_record(row, bases[join_id])
                        if record and (args.keep_ineligible or record["image_license"] in allowed):
                            output.append(record)
                    writer.write(output)
                    print(f"wrote GBIF images: {writer.rows_written}", end="\r")
            else:
                output = []
                for base in index.iter_values():
                    if base.get("image_url"):
                        record = finalize_record(base)
                        if args.keep_ineligible or record["image_license"] in allowed:
                            output.append(record)
                    if len(output) >= args.chunksize:
                        writer.write(output)
                        output = []
                writer.write(output)
            print(f"\nwrote {writer.rows_written} GBIF rows -> {args.out}")


if __name__ == "__main__":
    main()
