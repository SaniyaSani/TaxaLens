from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _clean_names(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        name = str(value or "").strip()
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            cleaned.append(name)
    return cleaned


class KeyNavigator:
    """Build a conservative, source-backed morphology checklist.

    The navigator does not reproduce a published dichotomous key and does not
    turn unchecked characters into a taxonomic decision.  It tells the user
    which evidence to acquire before comparing the model's candidate genera in
    an applicable key.
    """

    def __init__(self, guide_path: str | Path):
        self.guide_path = Path(guide_path)
        self.payload = json.loads(self.guide_path.read_text(encoding="utf-8"))

    def build(self, family: str | None, genera: Iterable[str] = ()) -> dict:
        family_name = str(family or "").strip()
        candidates = _clean_names(genera)[:5]
        families = self.payload.get("families", {})
        guide = families.get(family_name, self.payload.get("fallback", {}))
        coverage = "family-specific" if family_name in families else "general"
        return {
            "family": family_name or None,
            "title": guide.get("title", "Evidence checklist before using a genus key"),
            "scope": guide.get("scope", "adult Diptera"),
            "coverage": coverage,
            "candidate_genera": candidates,
            "candidate_cards": [
                {
                    "taxon": genus,
                    "instruction": "Record the character states below, then test this candidate in the cited key.",
                }
                for genus in candidates
            ],
            "steps": guide.get("steps", []),
            "stop_rules": guide.get("stop_rules", []),
            "sources": guide.get("sources", []),
            "disclaimer": self.payload.get("disclaimer", ""),
        }
