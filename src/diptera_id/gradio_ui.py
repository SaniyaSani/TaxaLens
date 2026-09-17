from __future__ import annotations

"""Pure formatting helpers for the public Gradio demo."""

import math
from typing import Any, Iterable

import pandas as pd


PREDICTION_COLUMNS = ["taxon", "score", "centroid similarity"]
NEIGHBOUR_COLUMNS = ["taxon", "similarity", "source", "license"]


def _number(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number * 100:.{digits}f}%"


def prediction_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    values = [
        {
            "taxon": row.get("taxon", ""),
            "score": _number(row.get("probability")),
            "centroid similarity": _number(row.get("centroid_similarity")),
        }
        for row in rows
    ]
    return pd.DataFrame(values, columns=PREDICTION_COLUMNS)


def neighbour_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    values = []
    for row in rows:
        taxon = row.get("species") or row.get("genus") or row.get("family") or "reference"
        values.append(
            {
                "taxon": taxon,
                "similarity": _number(row.get("similarity")),
                "source": row.get("source", ""),
                "license": row.get("photo_license") or row.get("license") or "",
            }
        )
    return pd.DataFrame(values, columns=NEIGHBOUR_COLUMNS)


def neighbour_gallery(rows: Iterable[dict[str, Any]]) -> list[tuple[str, str]]:
    gallery: list[tuple[str, str]] = []
    for row in rows:
        url = str(row.get("image_url") or "").strip()
        if not url.startswith(("https://", "http://")):
            continue
        taxon = row.get("species") or row.get("genus") or row.get("family") or "reference"
        gallery.append((url, f"{taxon} · {_number(row.get('similarity'))}"))
    return gallery


def accepted_markdown(payload: dict[str, Any]) -> str:
    accepted = payload.get("accepted") or {}
    open_set = payload.get("open_set") or {}
    if open_set.get("rejected"):
        rank = open_set.get("rank") or "known taxa"
        return f"### Unknown / review needed\nOpen-set gate stopped at **{rank}**."
    if accepted.get("rank") and accepted.get("taxon"):
        return (
            "### Conservative result\n"
            f"**{accepted['rank']} — {accepted['taxon']}**"
        )
    return "### Unknown / review needed\nNo rank passed the conservative threshold."


def guide_markdown(guide: dict[str, Any]) -> str:
    lines = [
        f"### {guide.get('title') or 'Morphology Key Navigator'}",
        f"**Scope:** {guide.get('scope') or 'check the source'}  ",
        f"**Coverage:** {guide.get('coverage') or 'general'}",
    ]
    candidates = guide.get("candidate_genera") or []
    if candidates:
        lines.append("\n**Compare in the key:** " + ", ".join(f"*{x}*" for x in candidates))
    for index, step in enumerate(guide.get("steps") or [], 1):
        title = step.get("title") or step.get("character") or f"Step {index}"
        instruction = (
            step.get("inspect")
            or step.get("instruction")
            or step.get("description")
            or ""
        )
        why = step.get("why") or ""
        lines.append(f"\n**{title}**  \n{instruction}")
        if why:
            lines.append(f"  \n*Why: {why}*")
    if guide.get("stop_rules"):
        lines.append("\n#### Stop rules")
        lines.extend(f"- {item}" for item in guide["stop_rules"])
    if guide.get("sources"):
        lines.append("\n#### Sources behind this checklist")
        for source in guide["sources"]:
            title = _escape_markdown(source.get("title") or "Reference")
            url = str(source.get("url") or "").strip()
            heading = f"[{title}]({url})" if url.startswith(("https://", "http://")) else title
            scope = source.get("scope") or "check scope before use"
            lines.append(f"- **{heading}** — {scope}")
    if guide.get("disclaimer"):
        lines.append("\n> " + str(guide["disclaimer"]))
    return "\n".join(lines)


def checklist_choices(guide: dict[str, Any]) -> list[str]:
    choices = []
    for index, step in enumerate(guide.get("steps") or [], 1):
        title = step.get("title") or step.get("character") or f"Step {index}"
        choices.append(str(title))
    return choices


def _escape_markdown(value: Any) -> str:
    return str(value or "").replace("[", "\\[").replace("]", "\\]")


def keys_markdown(payload: dict[str, Any]) -> str:
    rows = payload.get("results") or []
    if not rows:
        return "No applicable key lead found yet. Keep the family/genus candidates and verify morphology first."
    lines = ["### Published keys and revisions"]
    for row in rows:
        title = _escape_markdown(row.get("title") or "Untitled reference")
        url = str(row.get("url") or "").strip()
        heading = f"[{title}]({url})" if url.startswith(("https://", "http://")) else title
        meta = " · ".join(
            str(value)
            for value in (row.get("authors"), row.get("year"), row.get("provider"))
            if value
        )
        lines.append(f"\n- **{heading}**" + (f"  \n  {meta}" if meta else ""))
        if row.get("verification"):
            lines.append(f"  \n  *{_escape_markdown(row['verification'])}*")
    if payload.get("errors"):
        providers = sorted({str(item.get("provider")) for item in payload["errors"]})
        lines.append("\n_Search partly unavailable from: " + ", ".join(providers) + "._")
    return "\n".join(lines)
