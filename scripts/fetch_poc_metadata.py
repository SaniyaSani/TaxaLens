#!/usr/bin/env python3
"""Bounded, resumable PoC metadata through GBIF's public search API.

iNaturalist is restricted to its official Research-grade Observations dataset,
not relabelled museum data. Museum GBIF uses PRESERVED_SPECIMEN. This is a small
family-stratified convenience sample, not a random sample or a complete export.
No image bytes, whole-world archives, accounts, or download requests are used.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from diptera_id.corpus.io import ManifestWriter
from diptera_id.corpus.schema import OPEN_LICENSES, finalize_record, normalize_license
from ingest_gbif import occurrence_record
from recovery_support import atomic_json, backup_file, digest, nonempty

API = "https://api.gbif.org/v1"
INAT_DATASET = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"


class Client:
    def __init__(self, pause: float = 1.1, max_requests: int = 1500):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "TaxaLens/0.9.1 (bounded biodiversity research PoC)"
        self.pause, self.maximum, self.calls = pause, max_requests, 0

    def get(self, endpoint: str, params: dict | None = None) -> dict:
        for attempt in range(4):
            if self.calls >= self.maximum:
                raise RuntimeError("Request budget reached. Checkpoints retained; rerun later.")
            self.calls += 1
            time.sleep(self.pause)
            try:
                response = self.session.get(API + endpoint, params=params, timeout=(20, 90))
            except requests.exceptions.SSLError:
                raise  # Certificate errors need diagnosis, not repeated requests.
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as error:
                if attempt == 3:
                    raise RuntimeError(
                        "GBIF network request failed after 4 attempts. "
                        "Saved checkpoints retained; rerun this stage later."
                    ) from error
                delay = 5 * 2 ** attempt
                print(f"GBIF {type(error).__name__}: retry {attempt + 2}/4 in {delay}s; "
                      f"same endpoint={endpoint}, offset={(params or {}).get('offset', '-')}", flush=True)
                time.sleep(delay)
                continue
            if response.status_code in (401, 403):
                raise RuntimeError(f"Access refused ({response.status_code}); stopped without changing completed files.")
            if response.status_code == 429:
                # Do not attempt to evade throttling or keep hammering the server.
                raise RuntimeError(f"GBIF rate limit: retry later; Retry-After={response.headers.get('Retry-After', 'unspecified')}. Checkpoints saved.")
            if response.status_code >= 500 and attempt < 3:
                response.close()
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError("GBIF is temporarily unavailable")


def family_key(client: Client, family: str) -> int:
    item = client.get("/species/match", {"name": family, "rank": "FAMILY", "order": "Diptera"})
    if item.get("matchType") != "EXACT" or item.get("rank") != "FAMILY" or item.get("canonicalName", "").casefold() != family.casefold() or item.get("order") != "Diptera":
        raise RuntimeError(f"No exact Diptera family match for {family}; refusing a broader query")
    return int(item.get("acceptedUsageKey") or item["usageKey"])


def convert_occurrence(item: dict, source: str, family: str) -> dict | None:
    if item.get("order") != "Diptera" or item.get("family", "").casefold() != family.casefold():
        return None
    if source == "gbif" and item.get("basisOfRecord") != "PRESERVED_SPECIMEN":
        return None
    if source == "inat" and item.get("datasetKey") != INAT_DATASET:
        return None
    base = occurrence_record({**item, "gbifID": item.get("key", "")}, assume_filtered=True)
    if not base:
        return None
    record = base[1]
    for media in item.get("media", []):
        license_code = normalize_license(media.get("license", ""))
        url = str(media.get("identifier") or "")
        if license_code not in OPEN_LICENSES or not url.startswith(("http://", "https://")):
            continue
        if media.get("type", "StillImage") != "StillImage":
            continue
        record.update(image_url=url, image_license=license_code,
                      source_image_id=url, copyright_holder=media.get("rightsHolder", ""),
                      attribution=media.get("creator") or item.get("recordedBy", ""),
                      source_dataset=f"{item.get('datasetName', '')} [GBIF dataset {item.get('datasetKey', '')}]")
        if source == "inat":
            observation_url = str(item.get("occurrenceID") or item.get("references") or "")
            match = re.search(r"inaturalist\.org/observations/(\d+)", observation_url)
            if not match:
                continue
            obs_id = match.group(1)
            photo = re.search(r"/photos/(\d+)/", url)
            record.update(source="iNaturalist", source_record_id=obs_id, observation_id=obs_id,
                          source_url=observation_url, specimen_group_id=f"iNaturalist:{obs_id}",
                          basis_of_record="HUMAN_OBSERVATION", is_preserved_specimen=False,
                          label_quality="C" if record.get("species") else "D",
                          source_dataset=f"iNaturalist Research-grade Observations via GBIF:{INAT_DATASET}")
            if photo:
                record["source_image_id"] = photo.group(1)
                record["image_url"] = re.sub(r"/original\.", "/medium.", url)
        return finalize_record(record)  # At most one eligible photo per specimen.
    return None


def harvest(source: str, out: Path, families: list[str], limit: int, client: Client) -> dict:
    if not 0 < limit <= 50000 or not families or len(families) != len(set(families)):
        raise ValueError("Use 1..50000 records and a nonempty unique family list")
    cache = out.parent / (out.stem + "_checkpoints")
    cache.mkdir(parents=True, exist_ok=True)
    settings = {"source": source, "families": families, "limit": limit, "schema": 1,
                "delivery": "GBIF occurrence search", "sampling": "equal family quotas; one eligible image per observation/specimen"}
    settings_path = cache / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise RuntimeError("Checkpoint settings differ; choose a NEW output path. Existing data retained.")
    atomic_json(settings_path, settings)
    report_path = out.with_name(out.stem + ".report.json")
    if nonempty(out) and report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get("settings") == settings and report.get("complete") and report.get("sha256") == digest(out):
            print(f"Reusing completed {source}: {report['rows']} image records; no API requests", flush=True)
            return report
    if source == "inat":
        dataset = client.get("/dataset/" + INAT_DATASET)
        if "inaturalist" not in dataset.get("title", "").casefold():
            raise RuntimeError("iNaturalist dataset identity could not be verified")
    records, counts = [], {}
    for index, family in enumerate(families):
        quota = limit // len(families) + (index < limit % len(families))
        state_path = cache / f"family_{index:03d}.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"family": family, "offset": 0, "rows": [], "done": quota == 0}
        if state.get("family") != family:
            raise RuntimeError("Family checkpoint mismatch")
        if not state["done"] and "taxon_key" not in state:
            state["taxon_key"] = family_key(client, family)
            atomic_json(state_path, state)
        seen = {r["record_id"] for r in state["rows"]}
        while not state["done"] and len(state["rows"]) < quota:
            if state["offset"] >= 99000:
                state["done"] = True
                state["stop_reason"] = "search offset safety cap"
                break
            params = {"taxonKey": state["taxon_key"], "mediaType": "StillImage",
                      "limit": 200, "offset": state["offset"]}
            if source == "inat":
                params["datasetKey"] = INAT_DATASET
            else:
                params["basisOfRecord"] = "PRESERVED_SPECIMEN"
            payload = client.get("/occurrence/search", params)
            items = payload.get("results")
            if not isinstance(items, list):
                raise RuntimeError("Unexpected API response; checkpoint retained")
            page_ids = [r.get("key") for r in items]
            if items and page_ids == state.get("last_page_ids"):
                raise RuntimeError("API repeated a page; stopped to prevent a pagination loop")
            for item in items:
                record = convert_occurrence(item, source, family)
                if record and record["record_id"] not in seen:
                    state["rows"].append(record)
                    seen.add(record["record_id"])
                    if len(state["rows"]) >= quota:
                        break
            state.update(offset=state["offset"] + len(items), last_page_ids=page_ids,
                         done=not items or bool(payload.get("endOfRecords")) or len(state["rows"]) >= quota)
            atomic_json(state_path, state)
            print(f"{source} {family}: {len(state['rows'])}/{quota} eligible images; checked {state['offset']} specimens. Saved.", flush=True)
        atomic_json(state_path, state)
        records.extend(state["rows"])
        counts[family] = len(state["rows"])
    if not records:
        raise RuntimeError("No eligible image records found; existing manifest not replaced")
    pending = out.with_name(out.stem + ".pending.parquet")
    with ManifestWriter(pending) as writer:
        writer.write(records)
    backup_file(out)
    pending.replace(out)
    report = {"settings": settings, "rows": len(records), "requested": limit,
              "shortfall": max(0, limit - len(records)), "by_family": counts, "complete": True,
              "sha256": digest(out), "created_at": datetime.now(timezone.utc).isoformat(),
              "note": "Bounded convenience sample. Complete means harvest finished, NOT quota met. No image bytes downloaded."}
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["inat", "gbif"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=30000)
    parser.add_argument("--families-file", default=str(ROOT / "configs/target_diptera_families.json"))
    parser.add_argument("--max-requests", type=int, default=1500)
    args = parser.parse_args()
    families = json.loads(Path(args.families_file).read_text(encoding="utf-8"))["families"]
    harvest(args.source, Path(args.out), families, args.limit, Client(max_requests=args.max_requests))


if __name__ == "__main__":
    main()
