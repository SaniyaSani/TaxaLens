#!/usr/bin/env python3
"""TaxaLens v1.0: resumable, profile-driven four-source scale workflow.

The legacy v0.9 runner remains unchanged.  This entry point isolates every new
experiment under its own ``--data-root`` and derives BIOSCAN/checkpoint names
from the profile, so the same code can run the 200k, 500k and 1M gates.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_multisource_poc_v09 as legacy
from recovery_support import (
    atomic_json,
    backup_file,
    digest,
    nonempty,
    publish_bioscan,
    reusable_topup,
)
from run_bioscan import count_tag, validate_tag, verify_existing_selection
from select_bioscan_diptera import load_family_filter

EXPECTED_SOURCES = ("BIOSCAN-5M", "iNaturalist", "GBIF", "DiSSCo")
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
POC_API_LIMIT = 50_000


run = legacy.run
make_shards = legacy.make_shards
embedding_profiles = legacy.embedding_profiles
embedding_root = legacy.embedding_root
embedding_checkpoint_ready = legacy.embedding_checkpoint_ready
quality_gate_ready = legacy.quality_gate_ready
find_bioscan_metadata = legacy.find_bioscan_metadata


def resolve_profile_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_profile(value: str | Path) -> tuple[dict, Path]:
    path = resolve_profile_path(value)
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"profile not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON profile {path}: {exc}") from exc
    validate_profile(profile)
    return profile, path


def safe_component(value: object, label: str) -> str:
    text = str(value).strip()
    if not SAFE_COMPONENT.fullmatch(text) or text in {".", ".."}:
        raise SystemExit(
            f"{label} must be one safe path component "
            "(letters, digits, dot, underscore or hyphen)"
        )
    return text


def target_family_scope(profile: dict) -> tuple[Path, list[str]]:
    value = str(profile.get("target_families_file", "")).strip()
    if not value:
        raise SystemExit("scale profile must define target_families_file")
    path = resolve_profile_path(value)
    names, _ = load_family_filter(path)
    return path, names


def validate_profile(profile: dict) -> None:
    if not isinstance(profile, dict):
        raise SystemExit("profile must be a JSON object")
    run_settings = profile.get("run")
    if not isinstance(run_settings, dict):
        raise SystemExit("scale profile must define a run object")
    safe_component(run_settings.get("id", ""), "run.id")
    targets = profile.get("source_targets")
    if not isinstance(targets, dict) or set(targets) != set(EXPECTED_SOURCES):
        raise SystemExit(
            "source_targets must contain exactly: " + ", ".join(EXPECTED_SOURCES)
        )
    try:
        normalized_targets = {name: int(targets[name]) for name in EXPECTED_SOURCES}
        total = int(profile["total_images"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("total_images and source_targets must be integers") from exc
    if total < 1 or any(value < 1 for value in normalized_targets.values()):
        raise SystemExit("total_images and every source target must be positive")
    if sum(normalized_targets.values()) != total:
        raise SystemExit("source targets do not add up to total_images")
    if int(profile.get("shard_size", 0)) < 1:
        raise SystemExit("shard_size must be positive")
    expected_tag = count_tag(normalized_targets["BIOSCAN-5M"])
    try:
        configured_tag = validate_tag(str(run_settings.get("bioscan_tag", expected_tag)))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if configured_tag != expected_tag:
        raise SystemExit(
            f"run.bioscan_tag={configured_tag!r} does not match the "
            f"BIOSCAN target ({expected_tag})"
        )
    acquisition = profile.get("acquisition", {})
    if acquisition and not isinstance(acquisition, dict):
        raise SystemExit("acquisition must be a JSON object")
    acquisition_modes = {
        "iNaturalist": ({"bulk", "bounded_api"}, "bulk"),
        "GBIF": ({"download", "bounded_api"}, "download"),
        "DiSSCo": ({"api"}, "api"),
    }
    for source, (allowed, default) in acquisition_modes.items():
        mode = str(acquisition.get(source, default))
        if mode not in allowed:
            raise SystemExit(f"unsupported acquisition mode for {source}: {mode}")
        if mode == "bounded_api" and normalized_targets[source] > POC_API_LIMIT:
            raise SystemExit(
                f"{source} target {normalized_targets[source]:,} exceeds the "
                f"bounded API safety cap {POC_API_LIMIT:,}; use bulk/download mode"
            )
    target_family_scope(profile)


def bioscan_tag(profile: dict) -> str:
    target = int(profile["source_targets"]["BIOSCAN-5M"])
    try:
        return validate_tag(str(profile["run"].get("bioscan_tag", count_tag(target))))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def paths(
    data_root: str | Path,
    profile: dict,
    bioscan_root: str | Path | None = None,
    bioscan_manifest: str | Path | None = None,
    source_root: str | Path | None = None,
    image_root: str | Path | None = None,
) -> dict[str, Path]:
    validate_profile(profile)
    root = Path(data_root).expanduser().resolve()
    raw = Path(source_root).expanduser().resolve() if source_root else root
    images = Path(image_root).expanduser().resolve() if image_root else root / "images"
    bioscan = Path(bioscan_root).expanduser().resolve() if bioscan_root else raw / "bioscan"
    tag = bioscan_tag(profile)
    prefix = f"diptera_{tag}"
    dissco_tag = count_tag(int(profile["source_targets"]["DiSSCo"]))
    normalized_bioscan = (
        Path(bioscan_manifest).expanduser().resolve()
        if bioscan_manifest
        else bioscan / f"bioscan_{prefix}_manifest.parquet"
    )
    pipeline = root / "pipeline"
    models = root / "models"
    return {
        "root": root,
        "source_root": raw,
        "images": images,
        "identity": root / "run_identity.json",
        "bioscan": bioscan,
        "bioscan_selection": bioscan / f"{prefix}_selection.csv",
        "bioscan_selection_report": bioscan / f"{prefix}_selection_report.json",
        "bioscan_images": bioscan / f"{prefix}_images",
        "bioscan_downloaded": bioscan / f"{prefix}_downloaded.csv",
        "bioscan_existing_report": bioscan / f"{prefix}_existing_report.json",
        "bioscan_manifest": normalized_bioscan,
        "bioscan_topup_selection": bioscan / f"{prefix}_added_families_topup_selection.csv",
        "bioscan_topup_report": bioscan / f"{prefix}_added_families_topup_selection_report.json",
        "bioscan_topup_downloaded": bioscan / f"{prefix}_added_families_topup_downloaded.csv",
        "bioscan_topup_download_report": bioscan / f"{prefix}_added_families_topup_download_report.json",
        "bioscan_topup_progress": bioscan / f"{prefix}_added_families_topup_progress.json",
        "inat": raw / "inaturalist",
        "gbif": raw / "gbif",
        "dissco": raw / "dissco",
        "dissco_export": raw / "dissco" / f"diptera_{dissco_tag}.jsonl",
        "poc": pipeline,
        "corpus": pipeline / "corpus",
        "plan": pipeline / "training_plan.parquet",
        "cached_plan": pipeline / "training_plan_cached.parquet",
        "quality_plan": pipeline / "training_plan_quality.parquet",
        "quality_review": pipeline / "training_plan_quality_review.parquet",
        "quality_decisions": pipeline / "image_quality_decisions.parquet",
        "quality_report": pipeline / "image_quality_report.json",
        "evaluation_cohort": pipeline / "evaluation_cohort.parquet",
        "evaluation_cohort_report": pipeline / "evaluation_cohort_report.json",
        "quality_preview": pipeline / "image_quality_review_contact_sheet.jpg",
        "quality_overrides": pipeline / "image_quality_overrides.csv",
        "quality_checkpoints": pipeline / "image_quality_checkpoints",
        "quality_crops": root / "images_quality_crops",
        "shards": pipeline / "quality_manifest_shards",
        "embedded": pipeline / "quality_embedding_shards" / "dino",
        "embedded_bioclip": pipeline / "quality_embedding_shards" / "bioclip",
        "models": models,
        "models_dino": models / "comparisons" / "dino",
        "models_bioclip": models / "comparisons" / "bioclip",
    }


def source_config(layout: dict[str, Path], profile: dict) -> dict:
    config = legacy.source_config(layout)
    config["sources"]["dissco"]["input"] = str(layout["dissco_export"])
    pilot_rank = str(profile.get("pilot_rank", "family"))
    if pilot_rank not in {"family", "genus", "species"}:
        raise SystemExit(f"unsupported pilot_rank: {pilot_rank}")
    config["pilot"].update({
        # prepare_corpus also emits a small diagnostic pilot; the real scalable
        # plan is built later by plan_foundation_corpus with hierarchical rank.
        "rank": pilot_rank,
        "max_per_source_taxon": int(profile.get("max_per_source_taxon", 5_000)),
        "seed": int(profile.get("seed", 42)),
    })
    return config


def write_runtime_config(layout: dict[str, Path], profile: dict) -> Path:
    destination = layout["poc"] / "sources.runtime.json"
    atomic_json(destination, source_config(layout, profile))
    return destination


def run_identity(profile: dict, profile_path: Path, layout: dict[str, Path]) -> dict:
    return {
        "schema": 1,
        "run_id": str(profile["run"]["id"]),
        "profile": str(profile.get("profile", "")),
        "profile_path": str(profile_path),
        "profile_sha256": digest(profile_path),
        "total_images": int(profile["total_images"]),
        "source_targets": {
            source: int(profile["source_targets"][source]) for source in EXPECTED_SOURCES
        },
        "source_root": str(layout["source_root"]),
        "image_root": str(layout["images"]),
    }


def ensure_run_identity(
    layout: dict[str, Path], profile: dict, profile_path: Path, dry_run: bool
) -> None:
    expected = run_identity(profile, profile_path, layout)
    marker = layout["identity"]
    if marker.exists():
        try:
            saved = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid run identity marker: {marker}") from exc
        comparable = (
            "run_id", "profile_sha256", "total_images", "source_targets",
            "source_root", "image_root",
        )
        if any(saved.get(key) != expected.get(key) for key in comparable):
            raise SystemExit(
                "STOP: this data root belongs to a different profile revision. "
                f"Use a new --data-root instead of mixing experiments: {marker}"
            )
        return
    if dry_run:
        print(f"DRY RUN: would create run identity marker {marker}", flush=True)
        return
    atomic_json(marker, expected)


def doctor(layout: dict[str, Path], profile: dict) -> int:
    total_label = f"{int(profile['total_images']):,} training plan"
    family_file, family_names = target_family_scope(profile)
    checks = {
        "BIOSCAN selection": layout["bioscan_selection"],
        "BIOSCAN manifest": layout["bioscan_manifest"],
        "iNaturalist observations": layout["inat"] / "observations.csv.gz",
        "iNaturalist photos": layout["inat"] / "photos.csv.gz",
        "iNaturalist taxa": layout["inat"] / "taxa.csv.gz",
        "iNaturalist observers": layout["inat"] / "observers.csv.gz",
        "iNaturalist API alternative": layout["inat"] / "poc_api.parquet",
        "GBIF occurrence": layout["gbif"] / "extracted" / "occurrence.txt",
        "GBIF multimedia": layout["gbif"] / "extracted" / "multimedia.txt",
        "GBIF API alternative": layout["gbif"] / "poc_api.parquet",
        "DiSSCo export": layout["dissco_export"],
        total_label: layout["plan"],
        "cached image plan": layout["cached_plan"],
        "quality-approved plan": layout["quality_plan"],
        "quality report": layout["quality_report"],
        "frozen evaluation cohort": layout["evaluation_cohort"],
        "shard index": layout["shards"] / "shards.json",
        "trained classifiers": layout["models"] / "classifiers.joblib",
        "genus key catalog": ROOT / "data" / "key_catalog_v09.json",
        f"{len(family_names)}-family scope": family_file,
    }
    missing = 0
    print(f"Run: {profile['run']['id']} | raw target: {int(profile['total_images']):,}")
    print(f"Data root: {layout['root']}")
    print(f"Shared source root: {layout['source_root']}")
    print(f"Shared image cache: {layout['images']}")
    print(f"{'COMPONENT':<31} STATUS  PATH")
    print("-" * 110)
    for name, path in checks.items():
        ready = nonempty(path)
        missing += not ready
        print(f"{name:<31} {'ready' if ready else 'missing':<7} {path}")
    print("-" * 110)
    print(f"{len(checks) - missing}/{len(checks)} components ready")
    print("Doctor is read-only: missing inputs are expected before their stage runs.")
    return missing


def checkpoint_bioscan(layout: dict[str, Path]) -> None:
    candidates = [
        layout["bioscan_selection"],
        layout["bioscan_selection_report"],
        layout["bioscan_downloaded"],
        layout["bioscan_manifest"],
        layout["bioscan_topup_selection"],
        layout["bioscan_topup_report"],
        layout["bioscan_topup_downloaded"],
        layout["bioscan_topup_download_report"],
        layout["bioscan_topup_progress"],
    ]
    saved = []
    for path in candidates:
        backup = backup_file(path)
        if backup:
            saved.append({"original": str(path), "backup": str(backup), "sha256": digest(path)})
    atomic_json(layout["poc"] / "bioscan_recovery_checkpoint.json", {"files": saved})
    print(f"Checkpoint: {len(saved)} files backed up. Image folders were not changed.", flush=True)


def run_bioscan_stage(layout: dict[str, Path], profile: dict, dry_run: bool) -> None:
    target = int(profile["source_targets"]["BIOSCAN-5M"])
    settings = profile.get("bioscan", {})
    family_file, _ = target_family_scope(profile)
    command = [
        sys.executable,
        "scripts/run_bioscan.py",
        "--root", str(layout["bioscan"]),
        "--max-records", str(target),
        "--dataset-tag", bioscan_tag(profile),
        "--max-per-taxon", str(int(settings.get("max_per_taxon", 500))),
        "--min-rank", str(settings.get("min_rank", "family")),
        "--seed", str(int(profile.get("seed", 42))),
        "--families-file", str(family_file),
    ]
    run(command, dry_run)


def verify_bioscan_selection(layout: dict[str, Path], profile: dict) -> None:
    settings = profile.get("bioscan", {})
    _, target_families = target_family_scope(profile)
    verify_existing_selection(
        layout["bioscan_selection"],
        layout["bioscan_selection_report"],
        max_records=int(profile["source_targets"]["BIOSCAN-5M"]),
        max_per_taxon=int(settings.get("max_per_taxon", 500)),
        min_rank=str(settings.get("min_rank", "family")),
        seed=int(profile.get("seed", 42)),
        target_families=target_families,
    )


def reuse_existing_bioscan(
    layout: dict[str, Path], profile: dict, dry_run: bool
) -> None:
    if layout["bioscan_manifest"].exists():
        if layout["bioscan_selection"].exists():
            verify_bioscan_selection(layout, profile)
        print("BIOSCAN already normalized; reusing:", layout["bioscan_manifest"])
        return
    if layout["bioscan_selection"].exists():
        verify_bioscan_selection(layout, profile)
    if not layout["bioscan_downloaded"].exists():
        if not layout["bioscan_selection"].exists():
            target = int(profile["source_targets"]["BIOSCAN-5M"])
            settings = profile.get("bioscan", {})
            family_file, _ = target_family_scope(profile)
            run([
                sys.executable,
                "scripts/run_bioscan.py",
                "--root", str(layout["bioscan"]),
                "--max-records", str(target),
                "--dataset-tag", bioscan_tag(profile),
                "--max-per-taxon", str(int(settings.get("max_per_taxon", 500))),
                "--min-rank", str(settings.get("min_rank", "family")),
                "--seed", str(int(profile.get("seed", 42))),
                "--families-file", str(family_file),
                "--selection-only",
                "--skip-metadata-download",
            ], dry_run)
        run([
            sys.executable,
            "scripts/index_existing_bioscan.py",
            "--selection", str(layout["bioscan_selection"]),
            "--image-dir", str(layout["bioscan_images"]),
            "--out-manifest", str(layout["bioscan_downloaded"]),
            "--report", str(layout["bioscan_existing_report"]),
            "--allow-partial",
        ], dry_run)
    run([
        sys.executable,
        "scripts/ingest_bioscan.py",
        "--metadata", str(layout["bioscan_downloaded"]),
        "--out", str(layout["bioscan_manifest"]),
    ], dry_run)


def topup_bioscan_families(
    layout: dict[str, Path], profile: dict, dry_run: bool
) -> None:
    selection = layout["bioscan_selection"]
    images = layout["bioscan_images"]
    downloaded = layout["bioscan_downloaded"]
    normalized = layout["bioscan_manifest"]
    topup_selection = layout["bioscan_topup_selection"]
    topup_downloaded = layout["bioscan_topup_downloaded"]
    if not selection.exists() and not dry_run:
        raise SystemExit(f"BIOSCAN selection not found: {selection}; run --stage bioscan first")
    if selection.exists():
        verify_bioscan_selection(layout, profile)
    if not dry_run:
        checkpoint_bioscan(layout)
    if not downloaded.exists():
        run([
            sys.executable,
            "scripts/index_existing_bioscan.py",
            "--selection", str(selection),
            "--image-dir", str(images),
            "--out-manifest", str(downloaded),
            "--report", str(layout["bioscan_existing_report"]),
            "--allow-partial",
        ], dry_run)
    settings = profile.get("bioscan_family_topup", {})
    families = [
        str(value).strip()
        for value in settings.get("families", ["Muscidae", "Tachinidae"])
        if str(value).strip()
    ]
    minimum = int(settings.get("min_images_per_family", 1500))
    max_per_taxon = int(settings.get("max_per_taxon", 250))
    report_path = layout["bioscan_topup_report"]
    reuse = (
        reusable_topup(topup_selection, report_path, families, minimum)
        if topup_selection.exists() or report_path.exists()
        else False
    )
    if not reuse:
        metadata = (
            find_bioscan_metadata(layout["bioscan"])
            if not dry_run
            else layout["bioscan"] / "bioscan5m" / "metadata" / "csv" / "BIOSCAN_5M_Insect_Dataset_metadata.csv"
        )
        run([
            sys.executable,
            "scripts/select_bioscan_family_topup.py",
            "--metadata", str(metadata),
            "--base-selection", str(selection),
            "--image-dir", str(images),
            "--families", ",".join(families),
            "--min-per-family", str(minimum),
            "--max-per-taxon", str(max_per_taxon),
            "--out", str(topup_selection),
            "--report", str(report_path),
        ], dry_run)
    run([
        sys.executable,
        "scripts/download_bioscan_subset.py",
        "--selection", str(topup_selection),
        "--image-dir", str(images),
        "--out-manifest", str(topup_downloaded),
        "--report", str(layout["bioscan_topup_download_report"]),
        "--progress", str(layout["bioscan_topup_progress"]),
    ], dry_run)
    merged = downloaded.with_name(downloaded.stem + ".pending.csv")
    pending = normalized.with_name(normalized.stem + ".pending.parquet")
    run([
        sys.executable,
        "scripts/merge_bioscan_manifests.py",
        str(downloaded), str(topup_downloaded),
        "--out", str(merged),
    ], dry_run)
    run([
        sys.executable,
        "scripts/ingest_bioscan.py",
        "--metadata", str(merged),
        "--out", str(pending),
    ], dry_run)
    if not dry_run:
        publish_bioscan(merged, pending, downloaded, normalized)
    print(
        "BIOSCAN family top-up complete; existing images were preserved and "
        f"only missing rows for {', '.join(families)} were requested.",
        flush=True,
    )


def download_inaturalist(
    layout: dict[str, Path], profile: dict, args: argparse.Namespace
) -> None:
    required = [layout["inat"] / name for name in (
        "observations.csv.gz", "photos.csv.gz", "taxa.csv.gz", "observers.csv.gz"
    )]
    if all(nonempty(path) for path in required):
        print("Reusing complete iNaturalist bulk metadata", flush=True)
        return
    mode = str(profile.get("acquisition", {}).get("iNaturalist", "bulk"))
    target = int(profile["source_targets"]["iNaturalist"])
    if mode == "bounded_api":
        run([
            sys.executable,
            "scripts/fetch_poc_metadata.py",
            "--source", "inat",
            "--out", str(layout["inat"] / "poc_api.parquet"),
            "--limit", str(target),
            "--families-file", str(ROOT / profile["target_families_file"]),
        ], args.dry_run)
        return
    command = [
        sys.executable,
        "scripts/download_inat_metadata.py",
        "--out-dir", str(layout["inat"]),
    ]
    if args.inat_archive:
        command.extend(["--archive", str(Path(args.inat_archive).expanduser().resolve())])
    else:
        if not args.allow_inat_bulk_download:
            raise SystemExit(
                "STOP: scalable iNaturalist acquisition uses the complete official "
                "metadata bundle. Check scratch capacity, then pass "
                "--allow-inat-bulk-download (or provide --inat-archive)."
            )
        command.append("--allow-bulk-download")
    run(command, args.dry_run)


def download_gbif(
    layout: dict[str, Path], profile: dict, args: argparse.Namespace
) -> None:
    occurrence = layout["gbif"] / "extracted" / "occurrence.txt"
    multimedia = layout["gbif"] / "extracted" / "multimedia.txt"
    if nonempty(occurrence) and nonempty(multimedia):
        print("Reusing extracted GBIF download", flush=True)
        return
    mode = str(profile.get("acquisition", {}).get("GBIF", "download"))
    target = int(profile["source_targets"]["GBIF"])
    if mode == "bounded_api":
        run([
            sys.executable,
            "scripts/fetch_poc_metadata.py",
            "--source", "gbif",
            "--out", str(layout["gbif"] / "poc_api.parquet"),
            "--limit", str(target),
            "--families-file", str(ROOT / profile["target_families_file"]),
        ], args.dry_run)
        return
    if args.gbif_download_key:
        run([
            sys.executable,
            "scripts/gbif_download.py",
            "fetch", args.gbif_download_key,
            "--out", str(layout["gbif"] / "gbif_download.zip"),
            "--extract", str(layout["gbif"] / "extracted"),
        ], args.dry_run)
        return
    request = [
        sys.executable,
        "scripts/request_gbif_download.py",
        "--out", str(layout["gbif"] / "request.json"),
    ]
    if args.submit_gbif_request:
        request.append("--submit")
    run(request, args.dry_run)
    if not args.dry_run:
        if args.submit_gbif_request:
            print(
                "GBIF request submitted. Save its download key; after it succeeds, "
                "rerun this stage with --gbif-download-key KEY.",
                flush=True,
            )
        else:
            print(
                "GBIF request JSON was created but not submitted. Inspect it, then "
                "rerun with --submit-gbif-request and GBIF credentials in the environment.",
                flush=True,
            )


def download_stage(
    layout: dict[str, Path], profile: dict, args: argparse.Namespace
) -> None:
    if args.source in ("all", "inat"):
        download_inaturalist(layout, profile, args)
    if args.source in ("all", "gbif"):
        download_gbif(layout, profile, args)
    if args.source in ("all", "dissco"):
        run([
            sys.executable,
            "scripts/download_dissco.py",
            "--out", str(layout["dissco_export"]),
            "--max-records", str(int(profile["source_targets"]["DiSSCo"])),
            "--resume",
        ], args.dry_run)


def selected_manifests(
    layout: dict[str, Path], shard_index: int | None, max_shards: int, dry_run: bool
) -> list[Path]:
    if shard_index is not None and max_shards:
        raise SystemExit("use either --shard-index or --max-shards, not both")
    if shard_index is not None:
        if shard_index < 0:
            raise SystemExit("--shard-index must be non-negative")
        path = layout["shards"] / f"shard_{shard_index:05d}.parquet"
        if not path.exists() and not dry_run:
            raise SystemExit(f"embedding shard does not exist: {path}")
        return [path]
    manifests = sorted(layout["shards"].glob("shard_*.parquet"))
    if max_shards:
        manifests = manifests[:max_shards]
    if not manifests and dry_run:
        return [layout["shards"] / "shard_00000.parquet"]
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "doctor", "checkpoint", "prefetch", "bioscan", "bioscan-topup",
            "download", "ingest", "assemble", "plan", "cache", "quality",
            "freeze-eval", "embed", "train", "evaluate", "keys",
        ],
    )
    parser.add_argument("--source", choices=["all", "inat", "gbif", "dissco"], default="all")
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--source-root",
        help="Shared raw-source cache; defaults to --data-root",
    )
    parser.add_argument(
        "--image-root",
        help="Shared downloaded-image cache; defaults to DATA_ROOT/images",
    )
    parser.add_argument("--bioscan-root", help="Existing BIOSCAN raw folder; may be outside --data-root")
    parser.add_argument("--bioscan-manifest", help="Existing normalized BIOSCAN parquet")
    parser.add_argument("--reuse-existing-bioscan", action="store_true")
    parser.add_argument("--config", default="configs/multisource_scale_v10_raw200k.json")
    parser.add_argument("--allow-inat-bulk-download", action="store_true")
    parser.add_argument("--inat-archive", help="Already downloaded official iNaturalist metadata tar.gz")
    parser.add_argument("--submit-gbif-request", action="store_true")
    parser.add_argument("--gbif-download-key")
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument(
        "--encoder", choices=["all", "dino", "bioclip"], default="all",
        help="For --stage embed: run both encoders or only one",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--family", default="")
    parser.add_argument("--genera", default="")
    args = parser.parse_args()

    profile, profile_path = load_profile(args.config)
    layout = paths(
        args.data_root,
        profile,
        args.bioscan_root,
        args.bioscan_manifest,
        args.source_root,
        args.image_root,
    )
    if args.stage == "doctor":
        doctor(layout, profile)
        return
    for path in (layout["root"], layout["poc"], layout["models"]):
        path.mkdir(parents=True, exist_ok=True)
    ensure_run_identity(layout, profile, profile_path, args.dry_run)

    if args.stage == "checkpoint":
        if not args.dry_run:
            checkpoint_bioscan(layout)
        return
    if args.stage == "prefetch":
        run([
            sys.executable,
            "scripts/cache_scale_models.py",
            "--config", str(profile_path),
        ], args.dry_run)
        return
    if args.stage == "bioscan":
        if args.reuse_existing_bioscan:
            reuse_existing_bioscan(layout, profile, args.dry_run)
        else:
            run_bioscan_stage(layout, profile, args.dry_run)
        return
    if args.stage == "bioscan-topup":
        topup_bioscan_families(layout, profile, args.dry_run)
        return
    if args.stage == "download":
        download_stage(layout, profile, args)
        return

    runtime_config = write_runtime_config(layout, profile)
    if args.stage in {"ingest", "assemble"}:
        run([
            sys.executable,
            "scripts/prepare_corpus.py",
            "--config", str(runtime_config),
            "--stage", args.stage,
        ], args.dry_run)
        return

    if args.stage == "plan":
        master = layout["corpus"] / "master_manifest.parquet"
        if not master.exists() and not args.dry_run:
            raise SystemExit("master manifest missing; run --stage assemble first")
        run([
            sys.executable,
            "scripts/plan_foundation_corpus.py",
            "--input", str(master),
            "--config", str(profile_path),
            "--out", str(layout["plan"]),
            "--report", str(layout["poc"] / "training_plan_report.json"),
        ], args.dry_run)
        run([
            sys.executable,
            "scripts/corpus_report.py",
            "--input", str(layout["plan"]),
            "--out-json", str(layout["poc"] / "corpus_report.json"),
            "--out-md", str(layout["poc"] / "corpus_report.md"),
        ], args.dry_run)
        return

    if args.stage == "cache":
        if not layout["plan"].exists() and not args.dry_run:
            raise SystemExit("training plan missing; run --stage plan first")
        run([
            sys.executable,
            "scripts/download_images.py",
            "--manifest", str(layout["plan"]),
            "--out", str(layout["cached_plan"]),
            "--image-root", str(layout["images"]),
            "--max-images", "0",
            "--allow-unbounded",
        ], args.dry_run)
        return

    if args.stage == "quality":
        if not layout["cached_plan"].exists() and not args.dry_run:
            raise SystemExit("cached training plan missing; finish --stage cache first")
        quality = profile.get("image_quality", {})
        command = [
            sys.executable,
            "scripts/filter_image_quality.py",
            "--manifest", str(layout["cached_plan"]),
            "--out", str(layout["quality_plan"]),
            "--review-out", str(layout["quality_review"]),
            "--decisions-out", str(layout["quality_decisions"]),
            "--report", str(layout["quality_report"]),
            "--preview", str(layout["quality_preview"]),
            "--overrides", str(layout["quality_overrides"]),
            "--checkpoint-dir", str(layout["quality_checkpoints"]),
            "--crop-root", str(layout["quality_crops"]),
            "--model", str(quality.get("model", "google/owlv2-base-patch16-ensemble")),
            "--strict-sources", ",".join(quality.get("strict_sources", ["GBIF", "DiSSCo"])),
            "--prompts", ",".join(quality.get("prompts", [
                "a fly", "a pinned fly", "a mosquito", "a pinned mosquito", "a gnat", "a midge"
            ])),
            "--score-threshold", str(quality.get("score_threshold", 0.10)),
            "--min-image-side", str(quality.get("min_image_side", 224)),
            "--min-contrast-std", str(quality.get("min_contrast_std", 2.0)),
            "--min-entropy", str(quality.get("min_entropy", 1.0)),
            "--min-specimen-side", str(quality.get("min_specimen_side", 64)),
            "--min-specimen-area-ratio", str(quality.get("min_specimen_area_ratio", 0.0025)),
            "--crop-below-area-ratio", str(quality.get("crop_below_area_ratio", 0.35)),
            "--crop-padding", str(quality.get("crop_padding", 0.50)),
            "--batch-size", str(quality.get("batch_size", 2)),
            "--checkpoint-size", str(quality.get("checkpoint_size", 256)),
            "--max-processing-side", str(quality.get("max_processing_side", 4096)),
        ]
        required_sources = quality.get("required_sources")
        if required_sources:
            command.extend(["--required-sources", ",".join(required_sources)])
        run(command, args.dry_run)
        make_shards(layout, profile, layout["quality_plan"], args.dry_run)
        return

    if args.stage == "freeze-eval":
        if not layout["quality_plan"].exists() and not args.dry_run:
            raise SystemExit("quality-approved plan missing; run --stage quality first")
        run([
            sys.executable,
            "scripts/freeze_evaluation_cohort.py",
            "--input", str(layout["quality_plan"]),
            "--out", str(layout["evaluation_cohort"]),
            "--report", str(layout["evaluation_cohort_report"]),
        ], args.dry_run)
        return

    if args.stage == "embed":
        if not args.dry_run and not quality_gate_ready(layout):
            raise SystemExit(
                "STOP: a complete quality gate matching the current approved manifest "
                "is required. Run --stage quality first."
            )
        encoders = embedding_profiles(profile)
        if args.encoder != "all" and args.encoder not in encoders:
            raise SystemExit(f"encoder {args.encoder!r} is not configured")
        selected = encoders if args.encoder == "all" else {args.encoder: encoders[args.encoder]}
        manifests = selected_manifests(
            layout, args.shard_index, args.max_shards, args.dry_run
        )
        if not manifests and not args.dry_run:
            raise SystemExit("quality-approved shards missing; run --stage quality first")
        for name, embedding in selected.items():
            print(f"=== encoder: {name} ===", flush=True)
            for manifest in manifests:
                out = embedding_root(layout, name) / manifest.stem
                complete = out / "complete.json"
                force = complete.exists() and not embedding_checkpoint_ready(
                    complete, manifest, embedding
                )
                if complete.exists() and not force:
                    print(f"already complete: {name}/{manifest.stem}", flush=True)
                    continue
                command = [
                    sys.executable,
                    "scripts/process_embedding_shard.py",
                    "--manifest", str(manifest),
                    "--out-dir", str(out),
                    "--backend", str(embedding.get("backend", name)),
                    "--model", str(embedding["model"]),
                    "--image-size", str(embedding["image_size"]),
                    "--tile-grid", str(embedding.get("tile_grid", 1)),
                    "--batch-size", str(embedding["batch_size"]),
                    "--local-only",
                ]
                if not embedding.get("include_whole", True):
                    command.append("--no-whole")
                if force:
                    command.append("--force")
                run(command, args.dry_run)
        return

    if args.stage == "train":
        encoders = embedding_profiles(profile)
        if set(encoders) != {"dino", "bioclip"}:
            raise SystemExit("hybrid training requires configured dino and bioclip encoders")
        if not args.dry_run:
            if not quality_gate_ready(layout):
                raise SystemExit("STOP: current quality gate is incomplete; training is blocked")
            expected = sorted(layout["shards"].glob("shard_*.parquet"))
            missing = []
            for name, embedding in encoders.items():
                for manifest in expected:
                    complete = embedding_root(layout, name) / manifest.stem / "complete.json"
                    if not embedding_checkpoint_ready(complete, manifest, embedding):
                        missing.append(f"{name}/{manifest.name}")
            if not expected or missing:
                raise SystemExit(
                    "STOP: all DINO and BioCLIP shards must finish before training; "
                    f"missing: {len(missing)}"
                )
        training = profile["training"]
        for name, model_dir in (
            ("dino", layout["models_dino"]),
            ("bioclip", layout["models_bioclip"]),
        ):
            run([
                sys.executable,
                "scripts/merge_embedding_shards.py",
                "--shard-root", str(embedding_root(layout, name)),
                "--manifest-root", str(layout["shards"]),
                "--out-dir", str(model_dir),
            ], args.dry_run)
        run([
            sys.executable,
            "scripts/fuse_embedding_models.py",
            "--dino-dir", str(layout["models_dino"]),
            "--bioclip-dir", str(layout["models_bioclip"]),
            "--out-dir", str(layout["models"]),
        ], args.dry_run)
        train_args = [
            "--min-family", str(training["min_family"]),
            "--min-genus", str(training["min_genus"]),
            "--min-species", str(training["min_species"]),
            "--species-label-quality", str(training["species_label_quality"]),
            "--gate-quantile", str(training["gate_quantile"]),
            "--gate-margin", str(training["gate_margin"]),
        ]
        for model_dir in (
            layout["models_dino"], layout["models_bioclip"], layout["models"]
        ):
            run([
                sys.executable,
                "scripts/train_hierarchical.py",
                "--model-dir", str(model_dir),
                "--cohort-manifest", str(layout["models"] / "embedded_manifest.csv"),
                *train_args,
            ], args.dry_run)
        run([
            sys.executable,
            "scripts/build_retrieval_index.py",
            "--model-dir", str(layout["models"]),
        ], args.dry_run)
        return

    if args.stage == "evaluate":
        run([
            sys.executable,
            "scripts/render_evaluation_v09.py",
            "--report", str(layout["models"] / "training_report_hierarchical.json"),
            "--out", str(layout["models"] / "evaluation_by_source.md"),
        ], args.dry_run)
        run([
            sys.executable,
            "scripts/compare_encoder_reports.py",
            "--dino", str(layout["models_dino"] / "training_report_hierarchical.json"),
            "--bioclip", str(layout["models_bioclip"] / "training_report_hierarchical.json"),
            "--fusion", str(layout["models"] / "training_report_hierarchical.json"),
            "--out-json", str(layout["models"] / "encoder_comparison.json"),
            "--out-md", str(layout["models"] / "encoder_comparison.md"),
        ], args.dry_run)
        return

    if args.stage == "keys":
        if not args.family and not args.genera:
            raise SystemExit("--stage keys requires --family and/or --genera")
        run([
            sys.executable,
            "scripts/find_genus_keys.py",
            "--family", args.family,
            "--genera", args.genera,
        ], args.dry_run)


if __name__ == "__main__":
    main()
