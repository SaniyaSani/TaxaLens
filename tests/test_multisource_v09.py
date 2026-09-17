from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "scripts/run_multisource_poc_v09.py"
    spec = importlib.util.spec_from_file_location("runner_v09", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_v09_profile_has_exact_four_source_100k_allocation():
    profile = json.loads((ROOT / "configs/multisource_poc_v09.json").read_text(encoding="utf-8"))
    assert profile["source_targets"] == {"BIOSCAN-5M": 30000, "iNaturalist": 30000, "GBIF": 25000, "DiSSCo": 15000}
    assert sum(profile["source_targets"].values()) == 100000
    assert profile["required_families"] == ["Muscidae", "Tachinidae"]
    assert profile["bioscan_family_topup"] == {
        "families": ["Muscidae", "Tachinidae"],
        "min_images_per_family": 1500,
        "max_per_taxon": 250,
    }
    encoders = {item["name"]: item for item in profile["embedding"]["encoders"]}
    assert set(encoders) == {"dino", "bioclip"}
    assert encoders["dino"]["backend"] == "dino"
    assert encoders["bioclip"]["model"] == "hf-hub:imageomics/bioclip-2"


def test_v09_scope_has_18_microdiptera_plus_muscidae_and_tachinidae():
    scope = json.loads((ROOT / "configs/target_diptera_families.json").read_text(encoding="utf-8"))
    assert len(scope["families"]) == 20
    assert len(set(scope["families"])) == 20
    assert {"Muscidae", "Tachinidae"}.issubset(scope["families"])


def test_v09_is_one_whole_image_and_key_finder_is_automatic():
    profile = json.loads((ROOT / "configs/multisource_poc_v09.json").read_text(encoding="utf-8"))
    assert profile["embedding"]["tile_grid"] == 1
    assert profile["embedding"]["segmentation_required"] is False
    assert profile["key_finder"]["automatic_after_genus_candidates"] is True
    assert profile["image_quality"]["strict_sources"] == ["GBIF", "DiSSCo"]
    assert profile["image_quality"]["required_sources"] == ["BIOSCAN-5M", "iNaturalist", "GBIF"]
    assert profile["image_quality"]["model"].startswith("google/owlv2-")


def test_runtime_source_config_points_all_sources_into_data_root(tmp_path):
    runner = load_runner()
    layout = runner.paths(tmp_path)
    config = runner.source_config(layout)
    assert set(config["sources"]) == {"inat", "bioscan", "gbif", "dissco", "normalized_manifests"}
    assert str(tmp_path.resolve()) in config["sources"]["gbif"]["occurrence"]


def test_explicit_bioscan_root_reuses_existing_location(tmp_path):
    runner = load_runner()
    existing = tmp_path / "already_downloaded_bioscan"
    existing_manifest = tmp_path / "Foundation_v06" / "manifests" / "bioscan_raw.parquet"
    layout = runner.paths(tmp_path / "TaxaLensData", existing, existing_manifest)
    assert layout["bioscan"] == existing.resolve()
    assert layout["bioscan_manifest"] == existing_manifest.resolve()
    assert runner.source_config(layout)["sources"]["normalized_manifests"] == [
        str(existing_manifest.resolve()), str(layout["inat"] / "poc_api.parquet"),
        str(layout["gbif"] / "poc_api.parquet"),
    ]


def test_doctor_runs_without_downloading(tmp_path):
    completed = subprocess.run([sys.executable, str(ROOT / "scripts/run_multisource_poc_v09.py"), "--stage", "doctor", "--data-root", str(tmp_path)], cwd=ROOT, capture_output=True, text=True, check=True)
    assert "genus key catalog" in completed.stdout
    assert "GBIF occurrence" in completed.stdout


def test_v09_colab_uses_v09_runner_and_starts_with_doctor():
    notebook = json.loads((ROOT / "notebooks/TaxaLens_v0.9_Multisource_PoC_Colab.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "run_multisource_poc_v09.py --stage doctor" in text
    assert "--max-shards 1" in text
    assert "find_genus_keys.py" in text
    assert "--reuse-existing-bioscan" in text
    assert "--stage bioscan-topup" in text
    assert "stage('quality')" in text
    assert text.index("stage('quality')") < text.index("stage('embed','--max-shards','1')")


def test_quality_stage_uses_separate_checked_shards(tmp_path):
    runner = load_runner()
    layout = runner.paths(tmp_path)
    assert layout["quality_plan"].name == "training_plan_quality.parquet"
    assert layout["shards"].name == "quality_manifest_shards"
    assert layout["embedded"].name == "quality_embedding_shards"
    assert layout["embedded_bioclip"].name == "quality_embedding_shards_bioclip"
    assert layout["quality_crops"] != tmp_path / "images"


def test_embed_requires_quality_report_and_matching_shard_index(tmp_path):
    import pandas as pd
    import hashlib

    runner = load_runner()
    layout = runner.paths(tmp_path)
    layout["quality_plan"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"record_id": "one"}]).to_parquet(layout["quality_plan"], index=False)
    pd.DataFrame([{"record_id": "one"}]).to_parquet(layout["cached_plan"], index=False)
    layout["shards"].mkdir(parents=True, exist_ok=True)
    shard = layout["shards"] / "shard_00000.parquet"
    pd.DataFrame([{"record_id": "one"}]).to_parquet(shard, index=False)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    layout["quality_report"].write_text(json.dumps({
        "complete": True, "accepted_rows": 1,
        "input_manifest_sha256": sha(layout["cached_plan"]),
        "quality_manifest_sha256": sha(layout["quality_plan"]),
    }))
    (layout["shards"] / "shards.json").write_text(json.dumps({
        "input_sha256": runner.digest(layout["quality_plan"]), "rows": 1,
        "shards": [{"path": str(shard), "rows": 1, "sha256": sha(shard)}],
    }))
    assert runner.quality_gate_ready(layout)
    layout["quality_report"].write_text(json.dumps({"complete": True, "accepted_rows": 2}))
    assert not runner.quality_gate_ready(layout)
