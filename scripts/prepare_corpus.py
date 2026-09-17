#!/usr/bin/env python3
"""Run all enabled corpus adapters and assemble one deduplicated pilot manifest."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from recovery_support import nonempty


def resolve(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else ROOT / path)


def add_optional(command: list[str], flag: str, value: str | int | None) -> None:
    if value not in (None, "", 0, False):
        command.extend([flag, str(value)])


def run(command: list[str], dry_run: bool) -> None:
    print("+", shlex.join(command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def source_commands(config: dict, output_dir: Path) -> tuple[list[list[str]], list[str]]:
    commands: list[list[str]] = []
    outputs: list[str] = []
    sources = config.get("sources", {})

    inat = sources.get("inat", {})
    if inat.get("enabled"):
        out = str(output_dir / "inat_raw.parquet")
        command = [sys.executable, str(ROOT / "scripts/ingest_inat.py"), "--observations", resolve(inat["observations"]), "--out", out]
        for key, flag in (("photos", "--photos"), ("taxa", "--taxa"), ("observers", "--observers")):
            add_optional(command, flag, resolve(inat.get(key)))
        add_optional(command, "--max-observations", inat.get("max_observations"))
        add_optional(command, "--quality-grades", inat.get("quality_grades"))
        commands.append(command)
        outputs.append(out)

    bioscan = sources.get("bioscan", {})
    if bioscan.get("enabled"):
        out = str(output_dir / "bioscan_raw.parquet")
        command = [sys.executable, str(ROOT / "scripts/ingest_bioscan.py"), "--metadata", resolve(bioscan["metadata"]), "--out", out]
        add_optional(command, "--image-root", resolve(bioscan.get("image_root")))
        commands.append(command)
        outputs.append(out)

    gbif = sources.get("gbif", {})
    if gbif.get("enabled"):
        out = str(output_dir / "gbif_raw.parquet")
        command = [sys.executable, str(ROOT / "scripts/ingest_gbif.py"), "--occurrence", resolve(gbif["occurrence"]), "--out", out]
        add_optional(command, "--multimedia", resolve(gbif.get("multimedia")))
        commands.append(command)
        outputs.append(out)

    dissco = sources.get("dissco", {})
    if dissco.get("enabled"):
        out = str(output_dir / "dissco_raw.parquet")
        command = [sys.executable, str(ROOT / "scripts/ingest_dissco.py"), "--input", resolve(dissco["input"]), "--out", out]
        commands.append(command)
        outputs.append(out)

    for local in sources.get("normalized_manifests", []):
        outputs.append(resolve(local))
    return commands, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--stage", choices=["all", "ingest", "assemble"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(resolve(args.config))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(resolve(config.get("output_dir", "data/corpus")))
    output_dir.mkdir(parents=True, exist_ok=True)
    ingest_commands, manifests = source_commands(config, output_dir)
    if not args.dry_run:
        needed = list(config.get("sources", {}).get("normalized_manifests", []))
        if args.stage in {"all", "ingest"}:
            for command in ingest_commands:
                for flag in ("--observations", "--photos", "--taxa", "--observers", "--metadata", "--occurrence", "--multimedia", "--input"):
                    if flag in command:
                        needed.append(command[command.index(flag) + 1])
        else:
            needed.extend(manifests)
        missing = [str(resolve(p)) for p in dict.fromkeys(needed) if not nonempty(resolve(p))]
        dissco_path = config.get("sources", {}).get("dissco", {}).get("input")
        if dissco_path and config["sources"]["dissco"].get("enabled"):
            checkpoint = Path(resolve(dissco_path) + ".checkpoint.json")
            if checkpoint.exists() and not json.loads(checkpoint.read_text()).get("finished"):
                raise SystemExit("STOP: DiSSCo download checkpoint is unfinished. Rerun its download cell before ingest.")
        if missing:
            raise SystemExit("STOP: source preparation is not complete. No corpus was overwritten.\nMissing/empty:\n  "
                             + "\n  ".join(missing)
                             + "\nRun the corresponding download cells first (BIOSCAN top-up is independent).")
    if args.stage in {"all", "ingest"}:
        for command in ingest_commands:
            run(command, args.dry_run)
    if args.stage == "ingest":
        return
    if not manifests:
        raise SystemExit("Enable at least one source or list a normalized manifest")

    combined = str(output_dir / "combined_before_dedup.parquet")
    deduplicated = str(output_dir / "master_deduplicated.parquet")
    master = str(output_dir / "master_manifest.parquet")
    pilot = config.get("pilot", {})
    pilot_out = str(output_dir / "pilot_manifest.csv")

    run([sys.executable, str(ROOT / "scripts/build_master_manifest.py"), *manifests, "--out", combined], args.dry_run)
    run([sys.executable, str(ROOT / "scripts/deduplicate_records.py"), "--input", combined, "--out", deduplicated], args.dry_run)
    run([sys.executable, str(ROOT / "scripts/build_master_manifest.py"), deduplicated, "--out", master], args.dry_run)
    command = [
        sys.executable,
        str(ROOT / "scripts/sample_training_subset.py"),
        "--input", master,
        "--out", pilot_out,
        "--rank", str(pilot.get("rank", "family")),
        "--max-per-taxon", str(pilot.get("max_per_taxon", 250)),
        "--min-per-taxon", str(pilot.get("min_per_taxon", 8)),
        "--max-per-source-taxon", str(pilot.get("max_per_source_taxon", 150)),
        "--seed", str(pilot.get("seed", 42)),
    ]
    run(command, args.dry_run)
    print(f"ready for image download and embeddings: {pilot_out}")


if __name__ == "__main__":
    main()
