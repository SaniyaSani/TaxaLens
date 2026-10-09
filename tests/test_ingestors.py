from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def run_script(name: str, *args: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, args)], cwd=ROOT, check=True)


def test_inaturalist_bulk_ingest(tmp_path: Path):
    observations = tmp_path / "observations.csv"
    photos = tmp_path / "photos.csv"
    out = tmp_path / "inat.csv"
    pd.DataFrame([{
        "id": "10", "order": "Diptera", "family": "Syrphidae", "genus": "Eristalis",
        "species": "Eristalis tenax", "eventDate": "2026-08-01",
    }]).to_csv(observations, index=False)
    pd.DataFrame([{
        "coreid": "10", "id": "77", "type": "StillImage",
        "identifier": "https://static.inaturalist.org/photos/77/medium.jpg", "license": "CC BY 4.0",
    }]).to_csv(photos, index=False)
    run_script("ingest_inat.py", "--observations", observations, "--photos", photos, "--out", out)
    frame = pd.read_csv(out)
    assert len(frame) == 1
    assert frame.iloc[0]["species"] == "Eristalis tenax"
    assert frame.iloc[0]["image_license"] == "CC-BY"


def test_official_inaturalist_four_table_ingest(tmp_path: Path):
    taxa = tmp_path / "taxa.csv.gz"
    observations = tmp_path / "observations.csv.gz"
    photos = tmp_path / "photos.csv.gz"
    observers = tmp_path / "observers.csv.gz"
    out = tmp_path / "inat_official.csv"
    pd.DataFrame([
        {"taxon_id": "1", "ancestry": "", "rank": "kingdom", "name": "Animalia", "active": "true"},
        {"taxon_id": "47822", "ancestry": "1", "rank": "order", "name": "Diptera", "active": "true"},
        {"taxon_id": "100", "ancestry": "1/47822", "rank": "family", "name": "Syrphidae", "active": "true"},
        {"taxon_id": "101", "ancestry": "1/47822/100", "rank": "genus", "name": "Eristalis", "active": "true"},
        {"taxon_id": "102", "ancestry": "1/47822/100/101", "rank": "species", "name": "Eristalis tenax", "active": "true"},
        {"taxon_id": "999", "ancestry": "1", "rank": "order", "name": "Coleoptera", "active": "true"},
    ]).to_csv(taxa, sep="\t", index=False, compression="gzip")
    pd.DataFrame([
        {"observation_uuid": "obs-fly", "observer_id": "7", "latitude": "47.1", "longitude": "8.2", "taxon_id": "102", "quality_grade": "research", "observed_on": "2026-08-01"},
        {"observation_uuid": "obs-beetle", "observer_id": "7", "taxon_id": "999", "quality_grade": "research"},
    ]).to_csv(observations, sep="\t", index=False, compression="gzip")
    pd.DataFrame([
        {"photo_uuid": "p-1", "photo_id": "77", "observation_uuid": "obs-fly", "observer_id": "7", "extension": "jpg", "license": "cc-by"},
        {"photo_uuid": "p-2", "photo_id": "88", "observation_uuid": "obs-beetle", "observer_id": "7", "extension": "jpg", "license": "cc-by"},
    ]).to_csv(photos, sep="\t", index=False, compression="gzip")
    pd.DataFrame([{"observer_id": "7", "login": "diptera_expert", "name": "D. Expert"}]).to_csv(
        observers, sep="\t", index=False, compression="gzip"
    )
    run_script(
        "ingest_inat.py", "--observations", observations, "--photos", photos,
        "--taxa", taxa, "--observers", observers, "--out", out,
    )
    frame = pd.read_csv(out)
    assert len(frame) == 1
    assert frame.iloc[0]["order"] == "Diptera"
    assert frame.iloc[0]["family"] == "Syrphidae"
    assert frame.iloc[0]["genus"] == "Eristalis"
    assert frame.iloc[0]["species"] == "Eristalis tenax"
    assert frame.iloc[0]["observer"] == "diptera_expert"
    assert frame.iloc[0]["image_url"].endswith("/photos/77/medium.jpg")



def test_official_inaturalist_hymenoptera_ingest(tmp_path: Path):
    taxa = tmp_path / "taxa_hymenoptera.csv.gz"
    observations = tmp_path / "observations_hymenoptera.csv.gz"
    photos = tmp_path / "photos_hymenoptera.csv.gz"
    out = tmp_path / "inat_hymenoptera.csv"

    pd.DataFrame([
        {"taxon_id": "1", "ancestry": "", "rank": "kingdom",
         "name": "Animalia", "active": "true"},
        {"taxon_id": "47201", "ancestry": "1", "rank": "order",
         "name": "Hymenoptera", "active": "true"},
        {"taxon_id": "200", "ancestry": "1/47201", "rank": "family",
         "name": "Apidae", "active": "true"},
        {"taxon_id": "201", "ancestry": "1/47201/200", "rank": "genus",
         "name": "Bombus", "active": "true"},
        {"taxon_id": "202", "ancestry": "1/47201/200/201", "rank": "species",
         "name": "Bombus terrestris", "active": "true"},
    ]).to_csv(taxa, sep="\t", index=False, compression="gzip")

    pd.DataFrame([
        {"observation_uuid": "obs-bee", "observer_id": "8",
         "taxon_id": "202", "quality_grade": "research",
         "observed_on": "2026-08-02"},
        {"observation_uuid": "obs-other", "observer_id": "8",
         "taxon_id": "999", "quality_grade": "research"},
    ]).to_csv(observations, sep="\t", index=False, compression="gzip")

    pd.DataFrame([
        {"photo_uuid": "p-bee", "photo_id": "501",
         "observation_uuid": "obs-bee", "observer_id": "8",
         "extension": "jpg", "license": "cc-by"},
        {"photo_uuid": "p-other", "photo_id": "502",
         "observation_uuid": "obs-other", "observer_id": "8",
         "extension": "jpg", "license": "cc-by"},
    ]).to_csv(photos, sep="\t", index=False, compression="gzip")

    run_script(
        "ingest_inat.py",
        "--observations", observations,
        "--photos", photos,
        "--taxa", taxa,
        "--order", "Hymenoptera",
        "--out", out,
    )

    frame = pd.read_csv(out)

    assert len(frame) == 1
    assert frame.iloc[0]["source_record_id"] == "obs-bee"
    assert frame.iloc[0]["order"] == "Hymenoptera"
    assert frame.iloc[0]["family"] == "Apidae"
    assert frame.iloc[0]["genus"] == "Bombus"
    assert frame.iloc[0]["species"] == "Bombus terrestris"

def test_bioscan_ingest_preserves_dna(tmp_path: Path):
    metadata = tmp_path / "bioscan.csv"
    out = tmp_path / "bioscan_out.csv"
    pd.DataFrame([{
        "processid": "B-1", "order": "Diptera", "family": "Muscidae", "genus": "Musca",
        "species": "Musca domestica", "dna_barcode": "ACTG", "dna_bin": "BOLD:AAA0001",
        "image_url": "https://example.org/B-1.jpg",
    }]).to_csv(metadata, index=False)
    run_script("ingest_bioscan.py", "--metadata", metadata, "--out", out)
    frame = pd.read_csv(out)
    assert frame.iloc[0]["dna_barcode"] == "ACTG"
    assert frame.iloc[0]["label_quality"] == "A"
    assert frame.iloc[0]["image_license"] == "CC-BY"


def test_gbif_preserved_specimen_ingest(tmp_path: Path):
    occurrence = tmp_path / "occurrence.txt"
    media = tmp_path / "multimedia.txt"
    out = tmp_path / "gbif.csv"
    pd.DataFrame([{
        "gbifID": "900", "order": "Diptera", "basisOfRecord": "PRESERVED_SPECIMEN",
        "family": "Tachinidae", "genus": "Tachina", "species": "Tachina fera",
    }]).to_csv(occurrence, sep="\t", index=False)
    pd.DataFrame([{
        "coreid": "900", "type": "StillImage", "identifier": "https://example.org/900.jpg",
        "license": "https://creativecommons.org/licenses/by-sa/4.0/",
    }]).to_csv(media, sep="\t", index=False)
    run_script("ingest_gbif.py", "--occurrence", occurrence, "--multimedia", media, "--out", out)
    frame = pd.read_csv(out)
    assert frame.iloc[0]["basis_of_record"] == "PRESERVED_SPECIMEN"
    assert frame.iloc[0]["image_license"] == "CC-BY-SA"


def test_dissco_opends_nested_media(tmp_path: Path):
    source = tmp_path / "dissco.json"
    out = tmp_path / "dissco.csv"
    source.write_text(json.dumps([{
        "digitalSpecimenId": "20.5000/specimen-1",
        "order": "Diptera", "family": "Empididae", "genus": "Empis", "species": "Empis tessellata",
        "ods:hasMedia": [{
            "digitalMediaId": "media-1", "mediaType": "StillImage",
            "accessURI": "https://example.org/media-1.jpg", "license": "CC0",
        }],
    }]), encoding="utf-8")
    run_script("ingest_dissco.py", "--input", source, "--out", out)
    frame = pd.read_csv(out)
    assert len(frame) == 1
    assert frame.iloc[0]["source"] == "DiSSCo"
    assert frame.iloc[0]["family"] == "Empididae"


def test_deduplication_and_group_split(tmp_path: Path):
    combined = tmp_path / "combined.csv"
    deduplicated = tmp_path / "deduplicated.csv"
    master = tmp_path / "master.csv"
    pd.DataFrame([
        {
            "source": "iNaturalist", "source_record_id": "1", "source_image_id": "77",
            "image_url": "https://static.inaturalist.org/photos/77/medium.jpg", "image_license": "CC-BY",
            "order": "Diptera", "family": "Syrphidae", "species": "Eristalis tenax", "label_quality": "C",
        },
        {
            "source": "mirror", "source_record_id": "x", "source_image_id": "y",
            "image_url": "https://static.inaturalist.org/photos/77/original.jpg", "image_license": "CC-BY",
            "order": "Diptera", "family": "Syrphidae", "species": "Eristalis tenax", "label_quality": "D",
        },
    ]).to_csv(combined, index=False)
    run_script("deduplicate_records.py", "--input", combined, "--out", deduplicated)
    frame = pd.read_csv(deduplicated)
    assert len(frame) == 1
    assert str(frame.iloc[0]["duplicate_group_id"]).startswith("dup:")
    run_script("build_master_manifest.py", deduplicated, "--out", master)
    assigned = pd.read_csv(master)
    assert assigned.iloc[0]["split"] in {"train", "val", "test"}
    assert assigned.iloc[0]["split_group"] == assigned.iloc[0]["duplicate_group_id"]


def test_bioscan_official_holdout_is_preserved(tmp_path: Path):
    source = tmp_path / "bioscan.csv"
    master = tmp_path / "master.csv"
    pd.DataFrame([{
        "source": "BIOSCAN-5M", "source_record_id": "B-1", "source_image_id": "B-1",
        "local_path": "/tmp/B-1.jpg", "image_license": "CC-BY", "order": "Diptera",
        "family": "Muscidae", "source_split": "test_unseen",
    }]).to_csv(source, index=False)
    run_script("build_master_manifest.py", source, "--out", master)
    frame = pd.read_csv(master)
    assert frame.iloc[0]["split"] == "test"


def test_prepare_corpus_dry_run(tmp_path: Path):
    config = tmp_path / "pilot.json"
    config.write_text(json.dumps({
        "output_dir": str(tmp_path / "corpus"),
        "sources": {"normalized_manifests": [str(tmp_path / "normalized.csv")]},
        "pilot": {"rank": "family"},
    }), encoding="utf-8")
    run_script("prepare_corpus.py", "--config", config, "--dry-run")
