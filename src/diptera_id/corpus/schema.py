from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Iterable, Mapping


MASTER_COLUMNS = [
    "record_id",
    "source",
    "source_record_id",
    "source_image_id",
    "image_url",
    "local_path",
    "image_sha256",
    "image_license",
    "photo_license",  # backwards-compatible alias used by v0.1
    "copyright_holder",
    "attribution",
    "publisher",
    "source_dataset",
    "source_url",
    "basis_of_record",
    "is_preserved_specimen",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "subfamily",
    "tribe",
    "genus",
    "species",
    "subspecies",
    "original_scientific_name",
    "accepted_scientific_name",
    "taxon_id",
    "identification_level",
    "label_quality",
    "identified_by",
    "identification_remarks",
    "country",
    "state_province",
    "latitude",
    "longitude",
    "elevation_m",
    "event_date",
    "sex",
    "life_stage",
    "view_type",
    "parent_specimen_id",
    "view_count",
    "available_views",
    "dna_barcode",
    "dna_bin",
    "observation_id",  # backwards-compatible v0.1 fields
    "photo_id",
    "observation_url",
    "observer",
    "specimen_group_id",
    "duplicate_group_id",
    "split_group",
    "source_split",
    "split",
    "eligible_supervised",
    "exclusion_reason",
]

OPEN_LICENSES = {"CC0", "CC-BY", "CC-BY-SA", "PROJECT-OWNED"}

_LICENSE_ALIASES = {
    "cc0": "CC0",
    "cc-0": "CC0",
    "publicdomain": "CC0",
    "public-domain": "CC0",
    "pdm": "PDM",
    "cc-by": "CC-BY",
    "ccby": "CC-BY",
    "cc-by-sa": "CC-BY-SA",
    "ccbysa": "CC-BY-SA",
    "cc-by-nc": "CC-BY-NC",
    "ccbync": "CC-BY-NC",
    "cc-by-nc-sa": "CC-BY-NC-SA",
    "ccbyncsa": "CC-BY-NC-SA",
    "cc-by-nd": "CC-BY-ND",
    "ccbynd": "CC-BY-ND",
    "cc-by-nc-nd": "CC-BY-NC-ND",
    "ccbyncnd": "CC-BY-NC-ND",
    "project-owned": "PROJECT-OWNED",
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "na"} else text


def first(row: Mapping[str, Any], names: Iterable[str], default: str = "") -> str:
    """Return the first non-empty value among source-specific column aliases."""
    lower = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        value = clean(lower.get(name.lower()))
        if value:
            return value
    return default


def stable_id(*parts: Any, prefix: str = "") -> str:
    payload = "\x1f".join(clean(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}" if prefix else digest


def normalize_license(value: Any) -> str:
    text = clean(value)
    if not text:
        return "UNKNOWN"
    low = text.lower().strip().rstrip("/")
    if "creativecommons.org/publicdomain/zero" in low:
        return "CC0"
    if "creativecommons.org/publicdomain/mark" in low:
        return "PDM"
    match = re.search(r"creativecommons\.org/licenses/([a-z-]+)", low)
    if match:
        code = match.group(1)
        return _LICENSE_ALIASES.get(f"cc-{code}", f"CC-{code.upper()}")
    plain = re.sub(r"[^a-z0-9]+", "-", low).strip("-").replace("creative-commons-", "")
    plain = re.sub(r"-(?:1|2|3|4)(?:-0)?(?:-international|-unported)?$", "", plain)
    if plain in _LICENSE_ALIASES:
        return _LICENSE_ALIASES[plain]
    compact = re.sub(r"[^a-z0-9-]", "", low).replace("creative-commons", "")
    compact = re.sub(r"-\d(?:\.\d)?$", "", compact)
    return _LICENSE_ALIASES.get(compact, text.upper())


def canonical_taxon(value: Any, rank: str = "species") -> str:
    """Conservatively remove authorship while preserving a canonical binomial."""
    text = re.sub(r"\s+", " ", clean(value)).strip()
    if not text:
        return ""
    text = re.sub(r"\s+\([^)]*\)\s*$", "", text).strip()
    tokens = text.split()
    if rank == "genus":
        return tokens[0]
    if rank == "species" and len(tokens) >= 2:
        return " ".join(tokens[:2])
    if rank == "subspecies" and len(tokens) >= 3:
        return " ".join(tokens[:3])
    return text


def taxon_level(record: Mapping[str, Any]) -> str:
    for rank in ("subspecies", "species", "genus", "tribe", "subfamily", "family", "order"):
        if clean(record.get(rank)):
            return rank
    return "unidentified"


def as_bool(value: Any) -> bool:
    return clean(value).lower() in {"1", "true", "yes", "y", "preserved_specimen"}


def finalize_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Fill stable IDs and compatibility fields for one normalized image row."""
    record = {column: raw.get(column, "") for column in MASTER_COLUMNS}
    for key in MASTER_COLUMNS:
        if key not in {"is_preserved_specimen", "eligible_supervised"}:
            record[key] = clean(record[key])

    record["source"] = record["source"] or "unknown"
    record["order"] = canonical_taxon(record["order"], "order")
    record["family"] = canonical_taxon(record["family"], "family")
    record["genus"] = canonical_taxon(record["genus"], "genus")
    record["species"] = canonical_taxon(record["species"], "species")
    record["subspecies"] = canonical_taxon(record["subspecies"], "subspecies")
    if not record["species"] and record["subspecies"]:
        record["species"] = canonical_taxon(record["subspecies"], "species")
    if not record["genus"] and record["species"]:
        record["genus"] = record["species"].split()[0]
    record["image_license"] = normalize_license(record["image_license"] or record["photo_license"])
    record["photo_license"] = record["image_license"]
    record["basis_of_record"] = record["basis_of_record"].upper()
    record["is_preserved_specimen"] = as_bool(record["is_preserved_specimen"]) or record["basis_of_record"] == "PRESERVED_SPECIMEN"
    record["identification_level"] = record["identification_level"] or taxon_level(record)

    source_record_id = record["source_record_id"] or record["observation_id"]
    source_image_id = record["source_image_id"] or record["photo_id"] or record["image_url"] or record["local_path"]
    record["source_record_id"] = source_record_id
    record["source_image_id"] = source_image_id
    record["observation_id"] = record["observation_id"] or source_record_id
    record["photo_id"] = record["photo_id"] or source_image_id

    specimen_group = record["specimen_group_id"] or record["parent_specimen_id"] or record["split_group"]
    if not specimen_group:
        specimen_group = f"{record['source']}:{source_record_id}" if source_record_id else stable_id(record["source"], source_image_id, prefix="group")
    record["specimen_group_id"] = specimen_group
    record["split_group"] = record["duplicate_group_id"] or record["split_group"] or specimen_group
    record["record_id"] = record["record_id"] or stable_id(record["source"], source_record_id, source_image_id, prefix="img")

    if not record["accepted_scientific_name"]:
        record["accepted_scientific_name"] = record["subspecies"] or record["species"] or record["genus"] or record["family"]

    if record["eligible_supervised"] in ("", None):
        has_label = record["identification_level"] in {
            "family",
            "subfamily",
            "tribe",
            "genus",
            "species",
            "subspecies",
        }
        record["eligible_supervised"] = bool(has_label and record["image_license"] in OPEN_LICENSES)
    else:
        record["eligible_supervised"] = as_bool(record["eligible_supervised"])

    if not record["eligible_supervised"] and not record["exclusion_reason"]:
        if record["image_license"] not in OPEN_LICENSES:
            record["exclusion_reason"] = f"license:{record['image_license']}"
        elif record["identification_level"] == "unidentified":
            record["exclusion_reason"] = "missing_taxonomy"

    return record
