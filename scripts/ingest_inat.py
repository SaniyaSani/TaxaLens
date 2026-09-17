#!/usr/bin/env python3
"""Normalize the official iNaturalist Open Data tables into the master manifest.

The official monthly bundle contains tab-separated observations, photos, taxa and
observers tables (despite the ``.csv.gz`` suffix).  This adapter resolves the
separate taxonomy table, filters to descendants of Diptera, and performs the
large observation/photo join with a temporary SQLite index.

For compatibility with v0.2, already flattened DwC-A/CSV exports remain
supported when ``--taxa`` is omitted.
"""
from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diptera_id.corpus.io import ManifestWriter, iter_table
from diptera_id.corpus.join import RecordIndex, rows_from_chunk
from diptera_id.corpus.schema import canonical_taxon, finalize_record, first, normalize_license


INAT_DIPTERA_TAXON_ID = "47822"
RANKS = {"kingdom", "phylum", "class", "order", "family", "subfamily", "tribe", "genus", "species", "subspecies"}


def ancestry_ids(value: str) -> list[str]:
    """Parse both slash- and backslash-separated iNaturalist ancestry strings."""
    return [part for part in re.split(r"[\\/|,;\s]+", str(value).strip()) if part]


def iter_inat_table(path: str | Path, chunksize: int, official_tsv: bool) -> Iterator[pd.DataFrame]:
    if not official_tsv:
        yield from iter_table(path, chunksize)
        return
    yield from pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
        low_memory=False,
    )


def load_taxa(path: str | Path, chunksize: int) -> tuple[dict[str, tuple[list[str], str, str]], set[str]]:
    """Load the relatively small taxonomy table and locate Diptera descendants."""
    taxa: dict[str, tuple[list[str], str, str]] = {}
    diptera: set[str] = set()
    for chunk in iter_inat_table(path, chunksize, official_tsv=True):
        for row in chunk.to_dict(orient="records"):
            taxon_id = first(row, ["taxon_id", "id"])
            if not taxon_id:
                continue
            ancestry = ancestry_ids(first(row, ["ancestry", "ancestor_ids"]))
            rank = first(row, ["rank"]).lower()
            name = first(row, ["name", "scientific_name"])
            taxa[taxon_id] = (ancestry, rank, name)
            if taxon_id == INAT_DIPTERA_TAXON_ID or INAT_DIPTERA_TAXON_ID in ancestry:
                diptera.add(taxon_id)
    if INAT_DIPTERA_TAXON_ID not in taxa:
        raise SystemExit(f"taxa table does not contain iNaturalist Diptera taxon {INAT_DIPTERA_TAXON_ID}")
    return taxa, diptera


def taxonomy_resolver(taxa: dict[str, tuple[list[str], str, str]]):
    @lru_cache(maxsize=200_000)
    def resolve(taxon_id: str) -> dict[str, str]:
        node = taxa.get(taxon_id)
        if not node:
            return {}
        ancestry, _rank, _name = node
        result: dict[str, str] = {}
        for candidate in [*ancestry, taxon_id]:
            ancestor = taxa.get(candidate)
            if not ancestor:
                continue
            _ancestor_ids, rank, name = ancestor
            if rank in RANKS and name:
                result[rank] = name
        return result

    return resolve


def official_observation_record(
    row: dict,
    diptera_taxa: set[str],
    resolve_taxonomy,
    quality_grades: set[str],
) -> tuple[str, dict] | None:
    observation_uuid = first(row, ["observation_uuid", "uuid", "id"])
    taxon_id = first(row, ["taxon_id"])
    quality = first(row, ["quality_grade"]).lower().replace("_", " ")
    if not observation_uuid or taxon_id not in diptera_taxa:
        return None
    if quality_grades and quality not in quality_grades:
        return None
    taxonomy = resolve_taxonomy(taxon_id)
    species = canonical_taxon(taxonomy.get("species", ""), "species")
    return observation_uuid, {
        "source": "iNaturalist",
        "source_record_id": observation_uuid,
        "observation_id": observation_uuid,
        "source_url": f"https://www.inaturalist.org/observations/{observation_uuid}",
        "source_dataset": "iNaturalist Licensed Observation Images",
        "basis_of_record": "HUMAN_OBSERVATION",
        "kingdom": taxonomy.get("kingdom", ""),
        "phylum": taxonomy.get("phylum", ""),
        "class": taxonomy.get("class", ""),
        "order": taxonomy.get("order", "Diptera"),
        "family": taxonomy.get("family", ""),
        "subfamily": taxonomy.get("subfamily", ""),
        "tribe": taxonomy.get("tribe", ""),
        "genus": taxonomy.get("genus", ""),
        "species": species,
        "subspecies": taxonomy.get("subspecies", ""),
        "original_scientific_name": taxonomy.get("subspecies") or species or taxonomy.get("genus", ""),
        "taxon_id": taxon_id,
        "latitude": first(row, ["latitude", "lat"]),
        "longitude": first(row, ["longitude", "lon", "lng"]),
        "event_date": first(row, ["observed_on", "event_date"]),
        "observer": first(row, ["observer_id"]),
        "label_quality": "C" if species and quality == "research" else "D",
    }


def legacy_observation_record(row: dict, assume_diptera: bool) -> tuple[str, dict] | None:
    order = first(row, ["order", "taxon_order", "dwc:order"])
    if not assume_diptera and order.lower() != "diptera":
        return None
    join_id = first(row, ["id", "observation_id", "gbifid", "occurrenceid", "coreid", "uuid"])
    if not join_id:
        return None
    rank = first(row, ["taxonrank", "taxon_rank", "rank"]).lower()
    scientific = first(row, ["scientificname", "scientific_name", "taxon_name"])
    species = first(row, ["species", "taxon_species"])
    genus = first(row, ["genus", "taxon_genus"])
    if not species and rank in {"species", "subspecies", "variety"}:
        species = canonical_taxon(scientific, "species")
    if not genus and species:
        genus = species.split()[0]
    return join_id, {
        "source": "iNaturalist",
        "source_record_id": join_id,
        "observation_id": join_id,
        "order": order or "Diptera",
        "family": first(row, ["family", "taxon_family"]),
        "subfamily": first(row, ["subfamily", "taxon_subfamily"]),
        "tribe": first(row, ["tribe", "taxon_tribe"]),
        "genus": genus,
        "species": species,
        "subspecies": first(row, ["subspecies", "infraspecificepithet"]),
        "original_scientific_name": scientific,
        "taxon_id": first(row, ["taxon_id", "taxonid", "taxonkey"]),
        "identified_by": first(row, ["identified_by", "identifiedby"]),
        "country": first(row, ["country", "countrycode"]),
        "state_province": first(row, ["stateprovince", "state_province", "place_guess"]),
        "latitude": first(row, ["decimallatitude", "latitude", "lat"]),
        "longitude": first(row, ["decimallongitude", "longitude", "lon", "lng"]),
        "elevation_m": first(row, ["verbatimelevation", "elevation", "elevation_m"]),
        "event_date": first(row, ["eventdate", "observed_on", "observation_date"]),
        "sex": first(row, ["sex"]),
        "life_stage": first(row, ["lifestage", "life_stage"]),
        "observer": first(row, ["recordedby", "observer", "user_login", "user_name"]),
        "source_url": first(row, ["references", "uri", "observation_url"]),
        "source_dataset": first(row, ["datasetname", "dataset_name"], "iNaturalist bulk export"),
        "basis_of_record": first(row, ["basisofrecord", "basis_of_record"], "HUMAN_OBSERVATION"),
        "label_quality": "C" if species else "D",
        "image_url": first(row, ["image_url", "photo_url", "identifier", "media"]),
        "image_license": first(row, ["photo_license", "media_license", "license", "rights"]),
        "source_image_id": first(row, ["photo_id", "media_id"]),
    }


def media_record(row: dict, base: dict, image_size: str, observer: dict | None = None) -> dict | None:
    media_type = first(row, ["type", "mediatype", "format", "dc:type"]).lower()
    if media_type and not any(token in media_type for token in ("image", "stillimage", "jpeg", "jpg", "png", "webp")):
        return None
    photo_id = first(row, ["photo_id", "id", "mediaid"])
    extension = first(row, ["extension"], "jpg").lstrip(".")
    url = first(row, ["image_url", "photo_url", "accessuri", "contenturl", "identifier", "url"])
    if not url and photo_id:
        url = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{image_size}.{extension}"
    if not url.startswith(("http://", "https://")):
        return None
    result = dict(base)
    observer = observer or {}
    observer_label = first(observer, ["login", "name"]) or result.get("observer", "")
    result.update({
        "source_image_id": photo_id or first(row, ["identifier"], url),
        "photo_id": photo_id,
        "image_url": url,
        "image_license": first(row, ["license", "licence", "rights", "rightsuri"]),
        "observer": observer_label,
        "copyright_holder": first(row, ["rightsHolder", "rightsholder", "creator", "owner"]) or observer_label,
        "attribution": first(row, ["attribution", "creator", "title"]) or observer_label,
    })
    return finalize_record(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--photos", help="Official photos.csv.gz or a DwC-A multimedia table")
    parser.add_argument("--taxa", help="Official taxa.csv.gz; enables exact iNaturalist Open Data mode")
    parser.add_argument("--observers", help="Optional official observers.csv.gz for attribution")
    parser.add_argument("--out", default="data/corpus/inat_raw.parquet")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--licenses", default="CC0,CC-BY,CC-BY-SA")
    parser.add_argument("--quality-grades", default="research", help="Official grades, comma separated; empty keeps all")
    parser.add_argument("--image-size", choices=["original", "large", "medium", "small", "thumb", "square"], default="medium")
    parser.add_argument("--max-observations", type=int, default=0, help="Pilot cap after filtering; 0 means all")
    parser.add_argument("--keep-ineligible", action="store_true")
    parser.add_argument("--assume-diptera", action="store_true", help="Legacy flattened exports only")
    args = parser.parse_args()

    official = bool(args.taxa)
    if official and not args.photos:
        raise SystemExit("Official iNaturalist Open Data mode requires --photos and --taxa")
    allowed = {normalize_license(value) for value in args.licenses.split(",")}
    quality = {value.strip().lower().replace("_", " ") for value in args.quality_grades.split(",") if value.strip()}

    taxa: dict[str, tuple[list[str], str, str]] = {}
    diptera_taxa: set[str] = set()
    resolve_taxonomy = None
    if official:
        print("loading iNaturalist taxonomy…")
        taxa, diptera_taxa = load_taxa(args.taxa, args.chunksize)
        resolve_taxonomy = taxonomy_resolver(taxa)
        print(f"loaded {len(taxa)} taxa; {len(diptera_taxa)} Diptera taxa")

    with RecordIndex() as observations, RecordIndex() as observers:
        if official and args.observers:
            for chunk in iter_inat_table(args.observers, args.chunksize, official_tsv=True):
                observers.add(rows_from_chunk(chunk, lambda row: (
                    first(row, ["observer_id", "id"]),
                    {"login": first(row, ["login"]), "name": first(row, ["name"])},
                ) if first(row, ["observer_id", "id"]) else None))

        indexed = 0
        stop = False
        for chunk in iter_inat_table(args.observations, args.chunksize, official_tsv=official):
            batch = []
            for row in chunk.to_dict(orient="records"):
                item = (
                    official_observation_record(row, diptera_taxa, resolve_taxonomy, quality)
                    if official
                    else legacy_observation_record(row, args.assume_diptera)
                )
                if item:
                    batch.append(item)
                    if args.max_observations and indexed + len(batch) >= args.max_observations:
                        stop = True
                        break
            indexed += observations.add(batch)
            print(f"indexed Diptera observations: {indexed}", end="\r")
            if stop:
                break
        print(f"\nindexed Diptera observations: {indexed}")

        with ManifestWriter(args.out) as writer:
            if args.photos:
                for chunk in iter_inat_table(args.photos, args.chunksize, official_tsv=official):
                    raw_rows = chunk.to_dict(orient="records")
                    join_ids = [first(row, ["observation_uuid", "coreid", "observation_id", "occurrenceid", "gbifid", "parentid"]) for row in raw_rows]
                    bases = observations.get_many(join_ids)
                    observer_ids = [first(row, ["observer_id"]) for row in raw_rows]
                    observer_rows = observers.get_many(observer_ids) if args.observers else {}
                    output = []
                    for row, join_id, observer_id in zip(raw_rows, join_ids, observer_ids):
                        if join_id not in bases:
                            continue
                        record = media_record(row, bases[join_id], args.image_size, observer_rows.get(observer_id))
                        if record and (args.keep_ineligible or record["image_license"] in allowed):
                            output.append(record)
                    writer.write(output)
                    print(f"wrote iNaturalist images: {writer.rows_written}", end="\r")
            else:
                output = []
                for base in observations.iter_values():
                    record = media_record(base, base, args.image_size)
                    if record and (args.keep_ineligible or record["image_license"] in allowed):
                        output.append(record)
                    if len(output) >= args.chunksize:
                        writer.write(output)
                        output = []
                writer.write(output)
            if writer.rows_written == 0:
                raise SystemExit("STOP: no eligible iNaturalist image rows; previous manifest retained")
            print(f"\nwrote {writer.rows_written} iNaturalist image rows -> {args.out}")


if __name__ == "__main__":
    main()
