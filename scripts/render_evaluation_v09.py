#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(value: object) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render TaxaLens v0.9 source-wise metrics as Markdown")
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
    family = payload.get("family", {})
    lines = [
        "# TaxaLens v0.9 evaluation by source",
        "",
        "> These are measured held-out metrics. Engineering tests are not biological accuracy.",
        "> Genus and species metrics are conditional head diagnostics using the true parent taxon; they are not end-to-end deployed accuracy.",
        "",
        "## Family head",
        "",
        f"Overall top-1: **{pct(family.get('top1_accuracy'))}**  ",
        f"Overall top-5: **{pct(family.get('top5_accuracy'))}**",
        f"Test-only top-1: **{pct(family.get('by_dataset_split', {}).get('test', {}).get('top1_accuracy'))}**",
        "",
        "| Source | N | Top-1 | Top-5 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for source, row in sorted(family.get("by_source", {}).items()):
        lines.append(f"| {source} | {row.get('n', 0)} | {pct(row.get('top1_accuracy'))} | {pct(row.get('top5_accuracy'))} |")
    lines += ["", "## Conditional genus heads", ""]
    for family_name, row in sorted(payload.get("genus_by_family", {}).items()):
        lines.append(f"- **{family_name}** — {row.get('classes', 0)} genera; top-1 {pct(row.get('top1_accuracy'))}; top-5 {pct(row.get('top5_accuracy'))}")
    lines += [
        "",
        "## Interpretation guardrail",
        "",
        "A correct TaxaLens output may stop at family/genus or return unknown. Species suggestions require suitable evidence and an applicable taxonomic key.",
        "",
    ]
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
