from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "multisource_scale_v10_raw200k.json": (
        "v10_raw200k_swiss28",
        200_000,
        {"BIOSCAN-5M": 60_000, "iNaturalist": 60_000, "GBIF": 50_000, "DiSSCo": 30_000},
    ),
    "multisource_scale_v10_raw500k.json": (
        "v10_raw500k_swiss28",
        500_000,
        {"BIOSCAN-5M": 150_000, "iNaturalist": 150_000, "GBIF": 125_000, "DiSSCo": 75_000},
    ),
    "multisource_scale_v10_raw1m.json": (
        "v10_raw1m_swiss28",
        1_000_000,
        {"BIOSCAN-5M": 300_000, "iNaturalist": 300_000, "GBIF": 250_000, "DiSSCo": 150_000},
    ),
}


def load_module(filename: str, module_name: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def load_runner():
    return load_module("run_multisource_scale.py", "runner_scale_v10")


def load_bioscan_runner():
    return load_module("run_bioscan.py", "runner_bioscan_scale")


def profile(name: str = "multisource_scale_v10_raw200k.json") -> dict:
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_scale_profiles_define_the_three_monotonic_gates():
    previous = {source: 0 for source in next(iter(CONFIGS.values()))[2]}
    for name, (run_id, total, targets) in CONFIGS.items():
        payload = profile(name)
        assert payload["run"]["id"] == run_id
        assert payload["total_images"] == total
        assert payload["source_targets"] == targets
        assert sum(targets.values()) == total
        assert payload["acquisition"] == {
            "iNaturalist": "bulk",
            "GBIF": "download",
            "DiSSCo": "api",
        }
        assert all(targets[source] > previous[source] for source in targets)
        previous = targets


def test_v10_uses_versioned_swiss_28_scope_without_changing_v09():
    expected_additions = {
        "Culicidae",
        "Syrphidae",
        "Simuliidae",
        "Anthomyiidae",
        "Dolichopodidae",
        "Empididae",
        "Hybotidae",
        "Calliphoridae",
    }
    scope_path = ROOT / "configs/target_diptera_families_v10.json"
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    families = scope["families"]
    assert len(families) == 28
    assert len({family.casefold() for family in families}) == 28
    assert expected_additions <= set(families)
    assert scope["required_families"] == ["Muscidae", "Tachinidae"]

    for name in CONFIGS:
        assert profile(name)["target_families_file"] == (
            "configs/target_diptera_families_v10.json"
        )

    legacy = json.loads(
        (ROOT / "configs/target_diptera_families.json").read_text(encoding="utf-8")
    )
    assert len(legacy["families"]) == 20


def test_scale_paths_isolate_outputs_but_share_raw_and_image_caches(tmp_path):
    runner = load_runner()
    raw = tmp_path / "source_cache"
    images = tmp_path / "image_cache"
    layout = runner.paths(
        tmp_path / "run_200k",
        profile(),
        source_root=raw,
        image_root=images,
    )
    assert layout["poc"] == (tmp_path / "run_200k" / "pipeline").resolve()
    assert layout["models"] == (tmp_path / "run_200k" / "models").resolve()
    assert layout["inat"] == (raw / "inaturalist").resolve()
    assert layout["images"] == images.resolve()
    assert layout["bioscan_selection"].name == "diptera_60k_selection.csv"
    assert layout["bioscan_manifest"].name == "bioscan_diptera_60k_manifest.parquet"
    assert layout["bioscan_topup_selection"].name == "diptera_60k_added_families_topup_selection.csv"
    assert layout["dissco_export"].name == "diptera_30k.jsonl"
    assert runner.source_config(layout, profile())["pilot"]["rank"] == "family"
    assert "poc_v09" not in str(layout)
    assert "models_poc_v09" not in str(layout)


def test_scale_profile_rejects_mismatched_bioscan_tag_and_unsafe_run_id():
    runner = load_runner()
    payload = profile()
    payload["run"]["bioscan_tag"] = "30k"
    with pytest.raises(SystemExit, match="does not match"):
        runner.validate_profile(payload)

    payload = profile()
    payload["run"]["id"] = "../mixed-run"
    with pytest.raises(SystemExit, match="safe path component"):
        runner.validate_profile(payload)


def test_bioscan_runner_builds_dynamic_noncolliding_paths(tmp_path):
    bioscan = load_bioscan_runner()
    sixty = bioscan.dataset_paths(tmp_path, "60k")
    three_hundred = bioscan.dataset_paths(tmp_path, "300k")
    assert bioscan.count_tag(60_000) == "60k"
    assert bioscan.count_tag(1_000_000) == "1m"
    assert sixty["selection"].name == "diptera_60k_selection.csv"
    assert three_hundred["selection"].name == "diptera_300k_selection.csv"
    assert sixty["manifest"] != three_hundred["manifest"]


def test_bioscan_runner_refuses_reusing_a_tag_with_different_settings(tmp_path):
    bioscan = load_bioscan_runner()
    layout = bioscan.dataset_paths(tmp_path, "60k")
    layout["selection"].touch()
    layout["selection_report"].write_text(json.dumps({
        "requested": 30_000,
        "selected": 30_000,
        "max_per_taxon": 500,
        "min_rank": "family",
        "seed": 42,
        "target_families": [],
    }), encoding="utf-8")
    with pytest.raises(SystemExit, match="settings differ"):
        bioscan.verify_existing_selection(
            layout["selection"],
            layout["selection_report"],
            max_records=60_000,
            max_per_taxon=500,
            min_rank="family",
            seed=42,
        )


def test_bioscan_runner_refuses_a_changed_family_scope(tmp_path):
    bioscan = load_bioscan_runner()
    layout = bioscan.dataset_paths(tmp_path, "60k")
    layout["selection"].touch()
    layout["selection_report"].write_text(json.dumps({
        "requested": 60_000,
        "selected": 60_000,
        "max_per_taxon": 500,
        "min_rank": "family",
        "seed": 42,
        "target_families": ["Phoridae"],
    }), encoding="utf-8")
    with pytest.raises(SystemExit, match="settings differ"):
        bioscan.verify_existing_selection(
            layout["selection"],
            layout["selection_report"],
            max_records=60_000,
            max_per_taxon=500,
            min_rank="family",
            seed=42,
            target_families=["Phoridae", "Syrphidae"],
        )


def run_scale_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_multisource_scale.py"),
            "--config", "configs/multisource_scale_v10_raw200k.json",
            "--data-root", str(tmp_path / "run"),
            "--source-root", str(tmp_path / "sources"),
            "--image-root", str(tmp_path / "images"),
            *args,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def test_scale_doctor_is_read_only_with_count_specific_labels(tmp_path):
    completed = run_scale_cli(tmp_path, "--stage", "doctor")
    assert "Run: v10_raw200k_swiss28" in completed.stdout
    assert "200,000 training plan" in completed.stdout
    assert "diptera_60k_selection.csv" in completed.stdout
    assert "28-family scope" in completed.stdout
    assert "poc_v09" not in completed.stdout


def test_scale_bioscan_and_dissco_dry_runs_use_profile_targets(tmp_path):
    bioscan = run_scale_cli(tmp_path, "--stage", "bioscan", "--dry-run")
    assert "scripts/run_bioscan.py" in bioscan.stdout
    assert "--max-records 60000" in bioscan.stdout
    assert "--dataset-tag 60k" in bioscan.stdout
    assert "--families-file" in bioscan.stdout
    assert "target_diptera_families_v10.json" in bioscan.stdout

    dissco = run_scale_cli(
        tmp_path,
        "--stage", "download",
        "--source", "dissco",
        "--dry-run",
    )
    assert "scripts/download_dissco.py" in dissco.stdout
    assert "--max-records 30000" in dissco.stdout
    assert "diptera_30k.jsonl" in dissco.stdout


def test_one_embedding_shard_can_be_selected_for_a_slurm_array(tmp_path):
    runner = load_runner()
    layout = runner.paths(tmp_path / "run", profile())
    layout["shards"].mkdir(parents=True)
    for index in (0, 1, 2):
        (layout["shards"] / f"shard_{index:05d}.parquet").touch()
    selected = runner.selected_manifests(layout, shard_index=1, max_shards=0, dry_run=False)
    assert [path.name for path in selected] == ["shard_00001.parquet"]
    with pytest.raises(SystemExit, match="either"):
        runner.selected_manifests(layout, shard_index=1, max_shards=1, dry_run=False)


def test_run_identity_blocks_profile_changes_in_one_data_root(tmp_path):
    runner = load_runner()
    config = tmp_path / "profile.json"
    payload = profile()
    config.write_text(json.dumps(payload), encoding="utf-8")
    layout = runner.paths(tmp_path / "run", payload)
    layout["root"].mkdir(parents=True)
    runner.ensure_run_identity(layout, payload, config, dry_run=False)

    changed = copy.deepcopy(payload)
    changed["seed"] = 99
    config.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(SystemExit, match="different profile revision"):
        runner.ensure_run_identity(layout, changed, config, dry_run=False)


def test_freeze_evaluation_cohort_keeps_only_group_safe_val_and_test(tmp_path):
    source = tmp_path / "quality.parquet"
    out = tmp_path / "evaluation.parquet"
    report = tmp_path / "evaluation.json"
    pd.DataFrame([
        {"record_id": "a", "split_group": "g-a", "split": "train", "source": "GBIF", "family": "Muscidae"},
        {"record_id": "b", "split_group": "g-b", "split": "val", "source": "GBIF", "family": "Muscidae"},
        {"record_id": "c", "split_group": "g-c", "split": "test", "source": "BIOSCAN-5M", "family": "Tachinidae"},
    ]).to_parquet(source, index=False)
    command = [
        sys.executable,
        str(ROOT / "scripts/freeze_evaluation_cohort.py"),
        "--input", str(source),
        "--out", str(out),
        "--report", str(report),
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    frozen = pd.read_parquet(out)
    assert set(frozen["record_id"]) == {"b", "c"}
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["rows"] == 2
    assert payload["split_counts"] == {"val": 1, "test": 1}
    # An identical rerun is explicitly reusable and does not overwrite the cohort.
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
