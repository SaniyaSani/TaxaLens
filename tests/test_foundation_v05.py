from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from scripts.download_dissco import full_url, jsonapi_items, specimen_pid
from scripts.make_embedding_shards import shard_for
from scripts.plan_foundation_corpus import allocate_source_budgets, allocate_taxon_budgets


ROOT = Path(__file__).resolve().parents[1]


def test_source_budget_redistributes_unavailable_capacity():
    available = Counter({"iNaturalist": 100, "BIOSCAN-5M": 3, "GBIF": 100, "DiSSCo": 100})
    requested = {"iNaturalist": 10, "BIOSCAN-5M": 10, "GBIF": 10, "DiSSCo": 10}
    budget = allocate_source_budgets(available, requested, 40)
    assert budget["BIOSCAN-5M"] == 3
    assert sum(budget.values()) == 40


def test_source_budget_never_exceeds_total():
    available = Counter({"A": 100, "B": 100})
    budget = allocate_source_budgets(available, {"A": 90, "B": 90}, 50)
    assert sum(budget.values()) == 50


def test_taxon_budget_is_capped_and_exact():
    budget = allocate_taxon_budgets({"Rare": 3, "Common": 100, "Huge": 10000}, total=50, cap=30)
    assert sum(budget.values()) == 50
    assert budget["Rare"] == 3
    assert all(value <= 30 for value in budget.values())


def test_group_sharding_is_deterministic():
    assert shard_for("specimen-42", 17, 9) == shard_for("specimen-42", 17, 9)
    assert 0 <= shard_for("specimen-42", 17, 9) < 17


def test_dissco_jsonapi_helpers():
    item = {"id": "https://doi.org/20.5000.1025/ABC-123", "attributes": {"order": "Diptera"}}
    assert jsonapi_items({"data": [item]}) == [item]
    assert specimen_pid(item) == "20.5000.1025/ABC-123"
    assert full_url("https://example.test/api", specimen_pid(item)).endswith(
        "/digital-specimen/v1/20.5000.1025/ABC-123/full"
    )


def test_planner_requires_and_selects_all_four_sources(tmp_path):
    sources = ["iNaturalist", "BIOSCAN-5M", "GBIF", "DiSSCo"]
    rows = []
    for source in sources:
        for index in range(3):
            rows.append({
                "record_id": f"{source}-{index}",
                "source": source,
                "source_record_id": f"{source}-specimen-{index}",
                "source_image_id": str(index),
                "image_url": f"https://example.test/{source}/{index}.jpg",
                "image_license": "CC0",
                "order": "Diptera",
                "family": "Testidae",
                "eligible_supervised": True,
            })
    manifest = tmp_path / "master.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "profile": "test",
        "total_images": 8,
        "source_targets": {source: 2 for source in sources},
        "sampling_rank": "hierarchical",
        "max_per_source_taxon": 10,
        "seed": 1,
    }), encoding="utf-8")
    output = tmp_path / "plan.csv"
    report = tmp_path / "report.json"
    subprocess.run([
        sys.executable,
        str(ROOT / "scripts/plan_foundation_corpus.py"),
        "--input", str(manifest),
        "--config", str(profile),
        "--out", str(output),
        "--report", str(report),
    ], cwd=ROOT, check=True)
    result = pd.read_csv(output)
    assert len(result) == 8
    assert result.groupby("source").size().to_dict() == {source: 2 for source in sources}


def test_planner_full_mode_keeps_every_eligible_row(tmp_path):
    sources = ["iNaturalist", "BIOSCAN-5M", "GBIF", "DiSSCo"]
    rows = []
    for source in sources:
        for index in range(2):
            rows.append({
                "record_id": f"{source}-{index}", "source": source,
                "source_record_id": f"{source}-specimen-{index}", "source_image_id": str(index),
                "image_url": f"https://example.test/{source}/{index}.jpg", "image_license": "CC0",
                "order": "Diptera", "family": "Testidae", "eligible_supervised": True,
            })
    manifest = tmp_path / "master.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    profile = tmp_path / "full.json"
    profile.write_text(json.dumps({
        "profile": "full-test", "total_images": 0,
        "source_targets": {source: 1 for source in sources},
        "sampling_rank": "hierarchical", "max_per_source_taxon": 0, "seed": 1,
    }), encoding="utf-8")
    output = tmp_path / "full.csv"
    report = tmp_path / "full-report.json"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/plan_foundation_corpus.py"),
        "--input", str(manifest), "--config", str(profile), "--out", str(output), "--report", str(report),
    ], cwd=ROOT, check=True)
    assert len(pd.read_csv(output)) == len(rows)
    assert json.loads(report.read_text(encoding="utf-8"))["full_corpus"] is True


def test_plan_supports_source_specific_taxon_caps(tmp_path: Path):
    rows = []

    for source in ("BIOSCAN-5M", "GBIF"):
        for index in range(6):
            rows.append({
                "record_id": f"{source}-{index}",
                "source": source,
                "source_record_id": f"{source}-specimen-{index}",
                "source_image_id": str(index),
                "image_url": (
                    f"https://example.test/{source}/{index}.jpg"
                ),
                "image_license": "CC0",
                "order": "Hymenoptera",
                "family": "Scelionidae",
                "eligible_supervised": True,
            })

    manifest = tmp_path / "master.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    profile = tmp_path / "source-caps.json"
    profile.write_text(json.dumps({
        "profile": "source-cap-test",
        "total_images": 10,
        "source_targets": {
            "BIOSCAN-5M": 5,
            "GBIF": 5,
        },
        "sampling_rank": "hierarchical",
        "max_per_source_taxon": 2,
        "max_per_source_taxon_by_source": {
            "BIOSCAN-5M": 0,
        },
        "seed": 7,
    }), encoding="utf-8")

    output = tmp_path / "planned.csv"
    report_path = tmp_path / "report.json"

    subprocess.run([
        sys.executable,
        str(ROOT / "scripts/plan_foundation_corpus.py"),
        "--input", str(manifest),
        "--config", str(profile),
        "--out", str(output),
        "--report", str(report_path),
    ], cwd=ROOT, check=True)

    planned = pd.read_csv(output)
    selected = planned.groupby("source").size().to_dict()

    assert selected == {
        "BIOSCAN-5M": 5,
        "GBIF": 2,
    }

    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    assert report["max_per_source_taxon"] == 2
    assert report["max_per_source_taxon_by_source"] == {
        "BIOSCAN-5M": 0,
    }
