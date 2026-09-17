#!/usr/bin/env python3
"""TaxaLens v0.9: resumable four-source PoC from download through evaluation."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from recovery_support import nonempty, atomic_json, checkpoint_bioscan, reusable_topup, publish_bioscan, digest
from embedding_safety import checkpoint_ready


def run(command: list[str], dry_run: bool = False) -> None:
    print("+", shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True, stdin=subprocess.DEVNULL,
                       env={**os.environ, "PYTHONUNBUFFERED": "1"})


def paths(
    data_root: str | Path,
    bioscan_root: str | Path | None = None,
    bioscan_manifest: str | Path | None = None,
) -> dict[str, Path]:
    root = Path(data_root).expanduser().resolve()
    bioscan = Path(bioscan_root).expanduser().resolve() if bioscan_root else root / "bioscan"
    normalized_bioscan = (
        Path(bioscan_manifest).expanduser().resolve()
        if bioscan_manifest
        else (root / "manifests" / "bioscan_raw.parquet" if (root / "manifests" / "bioscan_raw.parquet").exists()
              else bioscan / "bioscan_diptera_30k_manifest.parquet")
    )
    poc = root / "poc_v09"
    return {
        "root": root,
        "bioscan": bioscan,
        "bioscan_manifest": normalized_bioscan,
        "inat": root / "inaturalist",
        "gbif": root / "gbif",
        "dissco": root / "dissco",
        "poc": poc,
        "corpus": poc / "corpus",
        "plan": poc / "training_plan.parquet",
        "cached_plan": poc / "training_plan_cached.parquet",
        "quality_plan": poc / "training_plan_quality.parquet",
        "quality_review": poc / "training_plan_quality_review.parquet",
        "quality_decisions": poc / "image_quality_decisions.parquet",
        "quality_report": poc / "image_quality_report.json",
        "quality_preview": poc / "image_quality_review_contact_sheet.jpg",
        "quality_overrides": poc / "image_quality_overrides.csv",
        "quality_checkpoints": poc / "image_quality_checkpoints",
        "quality_crops": root / "images_quality_crops",
        # Quality-derived paths are intentionally separate from legacy shards so
        # an interrupted migration can never mix unchecked and checked images.
        "shards": poc / "quality_manifest_shards",
        "embedded": poc / "quality_embedding_shards",
        "embedded_bioclip": poc / "quality_embedding_shards_bioclip",
        "models": root / "models_poc_v09",
        "models_dino": root / "models_poc_v09" / "comparisons" / "dino",
        "models_bioclip": root / "models_poc_v09" / "comparisons" / "bioclip",
    }


def source_config(layout: dict[str, Path]) -> dict:
    config = {
        "output_dir": str(layout["corpus"]),
        "sources": {
            "inat": {
                "enabled": True,
                "observations": str(layout["inat"] / "observations.csv.gz"),
                "photos": str(layout["inat"] / "photos.csv.gz"),
                "taxa": str(layout["inat"] / "taxa.csv.gz"),
                "observers": str(layout["inat"] / "observers.csv.gz"),
                "quality_grades": "research",
            },
            "bioscan": {"enabled": False},
            "gbif": {
                "enabled": True,
                "occurrence": str(layout["gbif"] / "extracted" / "occurrence.txt"),
                "multimedia": str(layout["gbif"] / "extracted" / "multimedia.txt"),
            },
            "dissco": {"enabled": True, "input": str(layout["dissco"] / "diptera_full.jsonl")},
            "normalized_manifests": [str(layout["bioscan_manifest"])],
        },
        "pilot": {"rank": "family", "max_per_taxon": 5000, "min_per_taxon": 8, "max_per_source_taxon": 5000, "seed": 42},
    }
    # Use existing official exports when complete; otherwise bounded PoC manifests.
    inat = config["sources"]["inat"]
    if not all(nonempty(inat[k]) for k in ("observations", "photos", "taxa", "observers")):
        inat["enabled"] = False
        config["sources"]["normalized_manifests"].append(str(layout["inat"] / "poc_api.parquet"))
    gbif = config["sources"]["gbif"]
    if not all(nonempty(gbif[k]) for k in ("occurrence", "multimedia")):
        gbif["enabled"] = False
        config["sources"]["normalized_manifests"].append(str(layout["gbif"] / "poc_api.parquet"))
    return config


def write_runtime_config(layout: dict[str, Path]) -> Path:
    destination = layout["poc"] / "sources_v09.runtime.json"
    atomic_json(destination, source_config(layout))
    return destination


def doctor(layout: dict[str, Path]) -> int:
    checks = {
        "BIOSCAN manifest": layout["bioscan_manifest"],
        "iNaturalist observations": layout["inat"] / "observations.csv.gz",
        "iNaturalist photos": layout["inat"] / "photos.csv.gz",
        "iNaturalist taxa (bulk)": layout["inat"] / "taxa.csv.gz",
        "iNaturalist observers (bulk)": layout["inat"] / "observers.csv.gz",
        "iNaturalist PoC (alternative)": layout["inat"] / "poc_api.parquet",
        "GBIF PoC (alternative)": layout["gbif"] / "poc_api.parquet",
        "BIOSCAN saved top-up": layout["bioscan"] / "diptera_added_families_topup_selection.csv",
        "GBIF occurrence": layout["gbif"] / "extracted" / "occurrence.txt",
        "GBIF multimedia": layout["gbif"] / "extracted" / "multimedia.txt",
        "DiSSCo export": layout["dissco"] / "diptera_full.jsonl",
        "100k training plan": layout["plan"],
        "cached image plan": layout["cached_plan"],
        "quality-approved plan": layout["quality_plan"],
        "image quality report": layout["quality_report"],
        "trained classifiers": layout["models"] / "classifiers.joblib",
        "genus key catalog": ROOT / "data" / "key_catalog_v09.json",
        "20-family scope": ROOT / "configs" / "target_diptera_families.json",
    }
    missing = 0
    print(f"{'COMPONENT':<28} STATUS  PATH")
    print("-" * 100)
    for name, path in checks.items():
        ready = nonempty(path)
        missing += not ready
        print(f"{name:<28} {'ready' if ready else 'missing':<7} {path}")
    print("-" * 100)
    print(f"{len(checks) - missing}/{len(checks)} components ready")
    print("Bulk tables and PoC manifests are alternatives; neither training nor downloads are started.")
    return missing


def make_shards(layout: dict[str, Path], profile: dict, manifest: Path, dry_run: bool) -> None:
    run([
        sys.executable, "scripts/make_embedding_shards.py", "--input", str(manifest),
        "--out-dir", str(layout["shards"]), "--shard-size", str(profile["shard_size"]),
        "--seed", str(profile["seed"]),
    ], dry_run)


def embedding_profiles(profile: dict) -> dict[str, dict]:
    embedding = profile["embedding"]
    configured = embedding.get("encoders")
    if configured:
        return {str(item["name"]): dict(item) for item in configured}
    return {"dino": {"name": "dino", "backend": "dino", **embedding}}


def embedding_root(layout: dict[str, Path], name: str) -> Path:
    return layout["embedded"] if name == "dino" else layout[f"embedded_{name}"]


def embedding_checkpoint_ready(complete: Path, manifest: Path, config: dict) -> bool:
    return checkpoint_ready(complete, manifest, {**config, "local_only": True})


def quality_gate_ready(layout: dict[str, Path]) -> bool:
    try:
        report = json.loads(layout["quality_report"].read_text(encoding="utf-8"))
        index = json.loads((layout["shards"] / "shards.json").read_text(encoding="utf-8"))
        shard_paths = sorted(layout["shards"].glob("shard_*.parquet"))
        indexed = index.get("shards", [])
        indexed_by_name = {Path(item["path"]).name: item for item in indexed}
        shards_match = (
            len(indexed_by_name) == len(indexed) == len(shard_paths)
            and all(
                path.name in indexed_by_name
                and int(indexed_by_name[path.name]["rows"]) >= 0
                and indexed_by_name[path.name]["sha256"] == digest(path)
                for path in shard_paths
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return (
        report.get("complete") is True
        and nonempty(layout["quality_plan"])
        and nonempty(layout["cached_plan"])
        and report.get("input_manifest_sha256") == digest(layout["cached_plan"])
        and report.get("quality_manifest_sha256") == digest(layout["quality_plan"])
        and index.get("input_sha256") == digest(layout["quality_plan"])
        and int(index.get("rows", -1)) == int(report.get("accepted_rows", -2))
        and sum(int(item["rows"]) for item in indexed) == int(index.get("rows", -1))
        and shards_match
    )


def reuse_existing_bioscan(layout: dict[str, Path], dry_run: bool) -> None:
    """Create the normalized BIOSCAN manifest from local files only."""
    root = layout["bioscan"]
    selection = root / "diptera_30k_selection.csv"
    downloaded = root / "diptera_30k_downloaded.csv"
    images = root / "diptera_30k_images"
    normalized = layout["bioscan_manifest"]
    if normalized.exists():
        print("BIOSCAN already normalized; reusing without download:", normalized)
        return
    if not downloaded.exists():
        if not selection.exists():
            run([
                sys.executable, "scripts/run_bioscan_30k.py", "--root", str(root),
                "--max-records", "30000", "--selection-only", "--skip-metadata-download",
            ], dry_run)
        run([
            sys.executable, "scripts/index_existing_bioscan.py",
            "--selection", str(selection), "--image-dir", str(images),
            "--out-manifest", str(downloaded),
            "--report", str(root / "diptera_30k_existing_report.json"),
            "--allow-partial",
        ], dry_run)
    run([
        sys.executable, "scripts/ingest_bioscan.py", "--metadata", str(downloaded),
        "--out", str(normalized),
    ], dry_run)


def find_bioscan_metadata(root: Path) -> Path:
    preferred = root / "bioscan5m" / "metadata" / "csv" / "BIOSCAN_5M_Insect_Dataset_metadata.csv"
    if preferred.exists():
        return preferred
    candidates = sorted(root.rglob("*metadata*.csv"))
    if not candidates:
        raise SystemExit(
            f"BIOSCAN metadata was not found under {root}; the family top-up will not download metadata automatically"
        )
    return candidates[0]


def topup_bioscan_families(layout: dict[str, Path], profile: dict, dry_run: bool) -> None:
    """Download only missing Muscidae/Tachinidae rows and preserve the existing 30k cache."""
    root = layout["bioscan"]
    selection = root / "diptera_30k_selection.csv"
    images = root / "diptera_30k_images"
    downloaded = root / "diptera_30k_downloaded.csv"
    normalized = layout["bioscan_manifest"]
    topup_selection = root / "diptera_added_families_topup_selection.csv"
    topup_downloaded = root / "diptera_added_families_topup_downloaded.csv"
    if not selection.exists() and not dry_run:
        raise SystemExit(f"existing BIOSCAN selection not found: {selection}")
    if not dry_run:
        checkpoint_bioscan(layout)
    if not downloaded.exists():
        run([
            sys.executable, "scripts/index_existing_bioscan.py",
            "--selection", str(selection), "--image-dir", str(images),
            "--out-manifest", str(downloaded),
            "--report", str(root / "diptera_30k_existing_report.json"),
            "--allow-partial",
        ], dry_run)
    settings = profile.get("bioscan_family_topup", {})
    families = [str(value).strip() for value in settings.get("families", ["Muscidae", "Tachinidae"]) if str(value).strip()]
    minimum = int(settings.get("min_images_per_family", 1500))
    max_per_taxon = int(settings.get("max_per_taxon", 250))
    report_path = root / "diptera_added_families_topup_selection_report.json"
    reuse = reusable_topup(topup_selection, report_path, families, minimum) if topup_selection.exists() or report_path.exists() else False
    if not reuse:
        metadata = find_bioscan_metadata(root) if not dry_run else root / "bioscan5m" / "metadata" / "csv" / "BIOSCAN_5M_Insect_Dataset_metadata.csv"
        run([
            sys.executable, "scripts/select_bioscan_family_topup.py",
            "--metadata", str(metadata), "--base-selection", str(selection),
            "--image-dir", str(images), "--families", ",".join(families),
            "--min-per-family", str(minimum), "--max-per-taxon", str(max_per_taxon),
            "--out", str(topup_selection),
            "--report", str(report_path),
        ], dry_run)
    run([
        sys.executable, "scripts/download_bioscan_subset.py",
        "--selection", str(topup_selection), "--image-dir", str(images),
        "--out-manifest", str(topup_downloaded),
        "--report", str(root / "diptera_added_families_topup_download_report.json"),
        "--progress", str(root / "diptera_added_families_topup_progress.json"),
    ], dry_run)
    merged = root / "diptera_30k_downloaded.pending.csv"
    pending = normalized.with_name(normalized.stem + ".pending.parquet")
    run([
        sys.executable, "scripts/merge_bioscan_manifests.py",
        str(downloaded), str(topup_downloaded), "--out", str(merged),
    ], dry_run)
    run([
        sys.executable, "scripts/ingest_bioscan.py", "--metadata", str(merged),
        "--out", str(pending),
    ], dry_run)
    if not dry_run:
        publish_bioscan(merged, pending, downloaded, normalized)
    print(
        "BIOSCAN family top-up complete: existing images were preserved; "
        f"only missing rows for {', '.join(families)} were requested."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["doctor", "checkpoint", "bioscan-topup", "download", "ingest", "assemble", "plan", "cache", "quality", "embed", "train", "evaluate", "keys"])
    parser.add_argument("--source", choices=["all", "inat", "gbif", "dissco"], default="all")
    parser.add_argument("--data-root", default="~/TaxaLensData")
    parser.add_argument("--bioscan-root", help="Existing BIOSCAN raw folder; may be outside --data-root")
    parser.add_argument(
        "--bioscan-manifest",
        help="Existing normalized BIOSCAN parquet, e.g. Foundation_v06/manifests/bioscan_raw.parquet",
    )
    parser.add_argument(
        "--reuse-existing-bioscan",
        action="store_true",
        help="Never download BIOSCAN; index and normalize only files already on disk",
    )
    parser.add_argument("--config", default="configs/multisource_poc_v09.json")
    parser.add_argument("--submit-gbif-request", action="store_true")
    parser.add_argument("--gbif-download-key")
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--encoder", choices=["all", "dino", "bioclip"], default="all",
                        help="For --stage embed: run both encoders or resume only one")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--family", default="")
    parser.add_argument("--genera", default="")
    args = parser.parse_args()

    layout = paths(args.data_root, args.bioscan_root, args.bioscan_manifest)
    profile_path = Path(args.config)
    if not profile_path.is_absolute():
        profile_path = ROOT / profile_path
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if sum(profile["source_targets"].values()) != profile["total_images"]:
        raise SystemExit("source targets do not add up to total_images")
    for path in (layout["root"], layout["poc"], layout["models"]):
        path.mkdir(parents=True, exist_ok=True)

    if args.stage == "doctor":
        doctor(layout)
        return

    if args.stage == "checkpoint":
        if not args.dry_run:
            checkpoint_bioscan(layout)
        return

    if args.stage == "bioscan-topup":
        topup_bioscan_families(layout, profile, args.dry_run)
        return

    if args.stage == "download":
        if args.reuse_existing_bioscan:
            reuse_existing_bioscan(layout, args.dry_run)
        elif not layout["bioscan_manifest"].exists():
            raise SystemExit("BIOSCAN manifest missing. To protect existing work, automatic BIOSCAN redownload is disabled. Pass --reuse-existing-bioscan to index saved images, or explicitly run run_bioscan_30k.py for a NEW dataset.")
        config = source_config(layout)
        for source, label in (("inat", "iNaturalist"), ("gbif", "GBIF")):
            if args.source not in ("all", source) or config["sources"][source]["enabled"]:
                continue
            if source == "gbif" and (args.gbif_download_key or args.submit_gbif_request):
                continue
            run([sys.executable, "scripts/fetch_poc_metadata.py", "--source", source,
                 "--out", str(layout[source] / "poc_api.parquet"),
                 "--limit", str(profile["source_targets"][label]),
                 "--families-file", str(ROOT / profile["target_families_file"])], args.dry_run)
        if args.source in ("all", "dissco"):
            run([sys.executable, "scripts/download_dissco.py", "--out", str(layout["dissco"] / "diptera_full.jsonl"), "--max-records", "15000", "--resume"], args.dry_run)
        if args.source in ("all", "gbif") and args.gbif_download_key:
            run([sys.executable, "scripts/gbif_download.py", "fetch", args.gbif_download_key, "--out", str(layout["gbif"] / "gbif_download.zip"), "--extract", str(layout["gbif"] / "extracted")], args.dry_run)
        elif args.source in ("all", "gbif") and args.submit_gbif_request:
            command = [sys.executable, "scripts/request_gbif_download.py", "--out", str(layout["gbif"] / "request.json")]
            if args.submit_gbif_request:
                command.append("--submit")
            run(command, args.dry_run)
            raise SystemExit("GBIF request submitted, but data are not ready yet. Check status, then rerun download with --gbif-download-key KEY. Do not run ingest yet.")
        return

    runtime_config = write_runtime_config(layout)
    if args.stage in {"ingest", "assemble"}:
        run([sys.executable, "scripts/prepare_corpus.py", "--config", str(runtime_config), "--stage", args.stage], args.dry_run)
        return

    if args.stage == "plan":
        master = layout["corpus"] / "master_manifest.parquet"
        if not master.exists() and not args.dry_run:
            raise SystemExit("master manifest missing; run --stage assemble first")
        run([sys.executable, "scripts/plan_foundation_corpus.py", "--input", str(master), "--config", str(profile_path), "--out", str(layout["plan"]), "--report", str(layout["poc"] / "training_plan_report.json")], args.dry_run)
        run([sys.executable, "scripts/corpus_report.py", "--input", str(layout["plan"]), "--out-json", str(layout["poc"] / "corpus_report.json"), "--out-md", str(layout["poc"] / "corpus_report.md")], args.dry_run)
        return

    if args.stage == "cache":
        if not layout["plan"].exists() and not args.dry_run:
            raise SystemExit("training plan missing; run --stage plan first")
        run([sys.executable, "scripts/download_images.py", "--manifest", str(layout["plan"]), "--out", str(layout["cached_plan"]), "--image-root", str(layout["root"] / "images"), "--max-images", "0", "--allow-unbounded"], args.dry_run)
        return

    if args.stage == "quality":
        if not layout["cached_plan"].exists() and not args.dry_run:
            raise SystemExit("cached training plan missing; finish --stage cache first")
        quality = profile.get("image_quality", {})
        command = [
            sys.executable, "scripts/filter_image_quality.py",
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
            "--prompts", ",".join(quality.get("prompts", ["a fly", "a pinned fly", "a mosquito", "a pinned mosquito", "a gnat", "a midge"])),
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

    if args.stage == "embed":
        if not args.dry_run and not quality_gate_ready(layout):
            raise SystemExit(
                "STOP: a complete quality gate matching the current approved manifest is required. "
                "Run --stage quality; cached JPEG files will be reused."
            )
        encoders = embedding_profiles(profile)
        if args.encoder != "all" and args.encoder not in encoders:
            raise SystemExit(f"encoder {args.encoder!r} is not configured")
        selected = encoders if args.encoder == "all" else {args.encoder: encoders[args.encoder]}
        manifests = sorted(layout["shards"].glob("shard_*.parquet"))
        if not manifests and not args.dry_run:
            raise SystemExit("quality-approved embedding shards missing; run --stage quality first")
        if args.max_shards:
            manifests = manifests[:args.max_shards]
        for name, embedding in selected.items():
            print(f"=== encoder: {name} ===", flush=True)
            for manifest in manifests:
                out = embedding_root(layout, name) / manifest.stem
                complete = out / "complete.json"
                force = complete.exists() and not embedding_checkpoint_ready(complete, manifest, embedding)
                if complete.exists() and not force:
                    continue
                command = [
                    sys.executable, "scripts/process_embedding_shard.py",
                    "--manifest", str(manifest), "--out-dir", str(out),
                    "--backend", embedding.get("backend", name),
                    "--model", embedding["model"],
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
                raise SystemExit("STOP: current quality gate is incomplete; training is blocked.")
            expected = sorted(layout["shards"].glob("shard_*.parquet"))
            missing = []
            for name, embedding in encoders.items():
                for manifest in expected:
                    complete = embedding_root(layout, name) / manifest.stem / "complete.json"
                    if not embedding_checkpoint_ready(complete, manifest, embedding):
                        missing.append(f"{name}/{manifest.name}")
            if not expected or missing:
                raise SystemExit(f"STOP: all DINO and BioCLIP shards must finish before training; missing: {len(missing)}. A one-shard smoke test is not the full corpus.")
        training = profile["training"]
        for name, model_dir in (("dino", layout["models_dino"]), ("bioclip", layout["models_bioclip"])):
            run([sys.executable, "scripts/merge_embedding_shards.py", "--shard-root", str(embedding_root(layout, name)), "--manifest-root", str(layout["shards"]), "--out-dir", str(model_dir)], args.dry_run)
        run([sys.executable, "scripts/fuse_embedding_models.py", "--dino-dir", str(layout["models_dino"]), "--bioclip-dir", str(layout["models_bioclip"]), "--out-dir", str(layout["models"])], args.dry_run)
        train_args = ["--min-family", str(training["min_family"]), "--min-genus", str(training["min_genus"]), "--min-species", str(training["min_species"]), "--species-label-quality", training["species_label_quality"], "--gate-quantile", str(training["gate_quantile"]), "--gate-margin", str(training["gate_margin"])]
        for model_dir in (layout["models_dino"], layout["models_bioclip"], layout["models"]):
            run([sys.executable, "scripts/train_hierarchical.py", "--model-dir", str(model_dir),
                 "--cohort-manifest", str(layout["models"] / "embedded_manifest.csv"), *train_args], args.dry_run)
        run([sys.executable, "scripts/build_retrieval_index.py", "--model-dir", str(layout["models"])], args.dry_run)
        return

    if args.stage == "evaluate":
        run([sys.executable, "scripts/render_evaluation_v09.py", "--report", str(layout["models"] / "training_report_hierarchical.json"), "--out", str(layout["models"] / "evaluation_by_source.md")], args.dry_run)
        run([
            sys.executable, "scripts/compare_encoder_reports.py",
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
        command = [sys.executable, "scripts/find_genus_keys.py", "--family", args.family, "--genera", args.genera]
        run(command, args.dry_run)


if __name__ == "__main__":
    main()
