#!/usr/bin/env python3
"""Compare DINO, BioCLIP and fused hierarchical reports on test specimens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def split_metrics(row: dict, split: str = "test") -> dict:
    return row.get("by_dataset_split", {}).get(split, {})


def aggregate(heads: dict, split: str = "test") -> dict:
    rows = [split_metrics(row, split) for row in heads.values()]
    rows = [row for row in rows if row.get("n", 0)]
    total = sum(int(row["n"]) for row in rows)
    if not total:
        return {"n": 0, "top1_accuracy": None, "top5_accuracy": None}
    return {
        "n": total,
        "top1_accuracy": sum(int(row["n"]) * float(row["top1_accuracy"]) for row in rows) / total,
        "top5_accuracy": sum(int(row["n"]) * float(row["top5_accuracy"]) for row in rows) / total,
    }


def summarize(report: dict) -> dict:
    return {
        "family_end_to_end": split_metrics(report.get("family", {})),
        "genus_conditional": aggregate(report.get("genus_by_family", {})),
        "species_conditional": aggregate(report.get("species_by_genus", {})),
    }


def pct(value: object) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino", required=True)
    parser.add_argument("--bioclip", required=True)
    parser.add_argument("--fusion", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()
    inputs = {"DINOv3": args.dino, "BioCLIP": args.bioclip, "DINO+BioCLIP": args.fusion}
    reports = {name: json.loads(Path(path).read_text(encoding="utf-8")) for name, path in inputs.items()}
    cohort_hashes = {name: report.get("evaluation_context", {}).get("cohort_sha256")
                     for name, report in reports.items()}
    if not all(cohort_hashes.values()) or len(set(cohort_hashes.values())) != 1:
        raise SystemExit("reports do not describe the same specimen cohort; comparison is blocked")
    result = {
        "evaluation_context": next(iter(reports.values()))["evaluation_context"],
        "models": {name: summarize(report) for name, report in reports.items()},
    }
    Path(args.out_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# DINOv3 vs BioCLIP vs fusion",
        "",
        "> Test specimens only. Genus/species rows are conditional diagnostics, not end-to-end accuracy.",
        "",
        "| Representation | Rank | N | Top-1 | Top-5 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    labels = {
        "family_end_to_end": "family (end-to-end)",
        "genus_conditional": "genus (conditional)",
        "species_conditional": "species (conditional)",
    }
    for model, ranks in result["models"].items():
        for key, row in ranks.items():
            lines.append(
                f"| {model} | {labels[key]} | {row.get('n', 0)} | "
                f"{pct(row.get('top1_accuracy'))} | {pct(row.get('top5_accuracy'))} |"
            )
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_md)


if __name__ == "__main__":
    main()
