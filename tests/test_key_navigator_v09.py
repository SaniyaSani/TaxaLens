from pathlib import Path

from diptera_id.key_navigator import KeyNavigator


ROOT = Path(__file__).resolve().parents[1]


def test_chironomidae_guide_is_stage_specific_and_keeps_candidates():
    navigator = KeyNavigator(ROOT / "data/key_navigator_v09.json")
    guide = navigator.build("Chironomidae", ["Chironomus", "Tanytarsus", "Chironomus"])
    assert guide["coverage"] == "family-specific"
    assert guide["candidate_genera"] == ["Chironomus", "Tanytarsus"]
    assert any(step["id"] == "hypopygium" for step in guide["steps"])
    assert any("species" in rule.lower() for rule in guide["stop_rules"])


def test_unknown_family_gets_safe_fallback():
    navigator = KeyNavigator(ROOT / "data/key_navigator_v09.json")
    guide = navigator.build("Unknownidae", ["Examplegenus"])
    assert guide["coverage"] == "general"
    assert guide["candidate_genera"] == ["Examplegenus"]
    assert len(guide["steps"]) >= 4
