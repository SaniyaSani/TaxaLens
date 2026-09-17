#!/usr/bin/env python3
"""Normalize DiSSCo openDS JSON/JSONL or a flattened DiSSCover export.

The connector accepts exported records instead of depending on one deployment's
search endpoint. Nested openDS media objects are expanded to one manifest row per
image.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diptera_id.corpus.io import ManifestWriter, iter_table
from diptera_id.corpus.schema import finalize_record, first, normalize_license


def terminal_name(key: str) -> str:
    return key.rsplit("/", 1)[-1].rsplit("#", 1)[-1].rsplit(":", 1)[-1].lower()


def values_for(obj: Any, names: set[str]) -> list[Any]:
    output: list[Any] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if terminal_name(str(key)) in names and not isinstance(value, (dict, list)):
                output.append(value)
            output.extend(values_for(value, names))
    elif isinstance(obj, list):
        for value in obj:
            output.extend(values_for(value, names))
    return output


def pick(obj: Any, *names: str) -> str:
    values = values_for(obj, {terminal_name(name) for name in names})
    for value in values:
        if str(value).strip():
            return str(value).strip()
    return ""


def media_objects(obj: Any) -> list[dict]:
    found: list[dict] = []

    def walk(value: Any, in_media: bool = False) -> None:
        if isinstance(value, dict):
            keys = {terminal_name(str(key)) for key in value}
            looks_like_media = bool(keys & {"accessuri", "contenturl", "mediatype", "format"}) or (
                in_media and bool(keys & {"identifier", "license", "rights"})
            )
            if looks_like_media:
                found.append(value)
                return
            for key, child in value.items():
                child_media = terminal_name(str(key)) in {"hasmedia", "media", "digitalmedia", "mediaobjects"}
                walk(child, in_media or child_media)
        elif isinstance(value, list):
            for child in value:
                walk(child, in_media)

    walk(obj)
    return found


def iter_objects(path: str | Path, chunksize: int) -> Iterator[dict]:
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".jsonl", ".ndjson")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    if name.endswith(".json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("results", "items", "data", "digitalSpecimens", "digitalSpecimen"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
            else:
                payload = [payload]
        yield from payload
        return
    for chunk in iter_table(path, chunksize):
        yield from chunk.to_dict(orient="records")


def specimen_rows(obj: dict, assume_diptera: bool) -> list[dict]:
    # /full wraps specimen, media and annotations separately. Never interpret
    # annotation history or specimen metadata links as image objects.
    envelope = obj.get("data", {})
    attributes = envelope.get("attributes", {}) if isinstance(envelope, dict) else {}
    if isinstance(attributes, dict) and "digitalSpecimen" in attributes:
        obj = {"id": envelope.get("id", ""), "source_url": envelope.get("id", ""),
               **attributes["digitalSpecimen"],
               "digitalMedia": [m.get("digitalMediaObject", m) for m in attributes.get("digitalMedia", []) if isinstance(m, dict)]}
    # Flattened exports use direct column lookup; nested openDS uses recursive lookup.
    order = first(obj, ["order", "ods:order"]) or pick(obj, "order")
    if not assume_diptera and order.lower() != "diptera":
        return []
    basis = pick(obj, "basisOfRecord")
    if basis and basis.replace("_", "").casefold() != "preservedspecimen":
        return []
    specimen_id = first(obj, ["id", "digital_specimen_id", "specimen_id", "doi"]) or pick(
        obj, "digitalSpecimenId", "physicalSpecimenId", "catalogNumber", "occurrenceID", "doi"
    )
    if not specimen_id:
        return []
    species = first(obj, ["species"]) or pick(obj, "species")
    base = {
        "source": "DiSSCo",
        "source_record_id": specimen_id,
        "basis_of_record": "PRESERVED_SPECIMEN",
        "is_preserved_specimen": True,
        "order": order or "Diptera",
        "family": first(obj, ["family"]) or pick(obj, "family"),
        "subfamily": first(obj, ["subfamily"]) or pick(obj, "subfamily"),
        "tribe": first(obj, ["tribe"]) or pick(obj, "tribe"),
        "genus": first(obj, ["genus"]) or pick(obj, "genus"),
        "species": species,
        "subspecies": first(obj, ["subspecies"]) or pick(obj, "subspecies", "infraspecificEpithet"),
        "original_scientific_name": first(obj, ["scientificname", "scientific_name"]) or pick(obj, "scientificName", "verbatimIdentification"),
        "taxon_id": first(obj, ["taxonid", "taxon_id"]) or pick(obj, "taxonID"),
        "identified_by": first(obj, ["identifiedby", "identified_by"]) or pick(obj, "identifiedBy"),
        "country": first(obj, ["country"]) or pick(obj, "country"),
        "state_province": first(obj, ["stateprovince", "state_province"]) or pick(obj, "stateProvince"),
        "latitude": first(obj, ["decimallatitude", "latitude"]) or pick(obj, "decimalLatitude", "latitude"),
        "longitude": first(obj, ["decimallongitude", "longitude"]) or pick(obj, "decimalLongitude", "longitude"),
        "event_date": first(obj, ["eventdate", "event_date"]) or pick(obj, "eventDate"),
        "sex": first(obj, ["sex"]) or pick(obj, "sex"),
        "life_stage": first(obj, ["lifestage", "life_stage"]) or pick(obj, "lifeStage"),
        "publisher": first(obj, ["publisher", "institutioncode"]) or pick(obj, "publisher", "institutionCode"),
        "source_dataset": first(obj, ["datasetname", "dataset_name"]) or pick(obj, "datasetName", "sourceSystemName"),
        "source_url": first(obj, ["source_url", "references"]) or pick(obj, "sourceURL", "landingPage", "doi"),
        "specimen_group_id": f"DiSSCo:{specimen_id}",
        "label_quality": "B" if species else "D",
    }

    media = media_objects(obj)
    if not media:
        # Also support already flattened one-row-per-media exports.
        media = [obj]
    rows = []
    for item in media:
        media_type = first(item, ["type", "mediatype", "format"]) or pick(item, "type", "mediaType", "format")
        if media_type and not any(token in media_type.lower() for token in ("image", "stillimage", "jpeg", "jpg", "png", "webp")):
            continue
        url = (first(item, ["accessuri", "contenturl", "image_url", "url"])
               or pick(item, "accessURI") or pick(item, "contentUrl") or pick(item, "downloadURL")
               or first(item, ["identifier"]) or pick(item, "identifier"))
        if not url.startswith(("http://", "https://")) or re.match(r"https?://(?:dx\.)?doi\.org/", url):
            continue
        record = dict(base)
        record.update({
            "source_image_id": first(item, ["media_id", "id", "@id"]) or pick(item, "digitalMediaId", "mediaID") or url,
            "image_url": url,
            "image_license": first(item, ["license", "licence", "rights"]) or pick(item, "license", "rights", "rightsURI"),
            "copyright_holder": first(item, ["rightsholder", "copyright_holder"]) or pick(item, "rightsHolder", "creator"),
            "attribution": first(item, ["attribution", "creator", "title"]) or pick(item, "creator", "title"),
        })
        rows.append(finalize_record(record))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="openDS JSON/JSONL or flattened CSV/TSV/Parquet export")
    parser.add_argument("--out", default="data/corpus/dissco_raw.parquet")
    parser.add_argument("--chunksize", type=int, default=20_000)
    parser.add_argument("--licenses", default="CC0,CC-BY,CC-BY-SA")
    parser.add_argument("--keep-ineligible", action="store_true")
    parser.add_argument("--assume-diptera", action="store_true")
    args = parser.parse_args()

    allowed = {normalize_license(value) for value in args.licenses.split(",")}
    buffer: list[dict] = []
    with ManifestWriter(args.out) as writer:
        for obj in iter_objects(args.input, args.chunksize):
            for record in specimen_rows(obj, args.assume_diptera):
                if args.keep_ineligible or record["image_license"] in allowed:
                    buffer.append(record)
            if len(buffer) >= args.chunksize:
                writer.write(buffer)
                buffer = []
                print(f"wrote DiSSCo images: {writer.rows_written}", end="\r")
        writer.write(buffer)
        if writer.rows_written == 0:
            raise SystemExit("STOP: DiSSCo export has no eligible image rows. Previous manifest retained; inspect the export/filters, do not run assemble.")
        print(f"\nwrote {writer.rows_written} DiSSCo rows -> {args.out}")


if __name__ == "__main__":
    main()
