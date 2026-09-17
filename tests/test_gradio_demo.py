from __future__ import annotations

import pandas as pd

from diptera_id.gradio_ui import (
    accepted_markdown,
    checklist_choices,
    guide_markdown,
    keys_markdown,
    neighbour_frame,
    prediction_frame,
)


def test_prediction_and_neighbour_tables_are_human_readable():
    predictions = prediction_frame([
        {"taxon": "Chironomidae", "probability": 0.934, "centroid_similarity": 0.812}
    ])
    assert isinstance(predictions, pd.DataFrame)
    assert predictions.iloc[0].to_dict() == {
        "taxon": "Chironomidae",
        "score": "93.4%",
        "centroid similarity": "81.2%",
    }

    neighbours = neighbour_frame([
        {"genus": "Chironomus", "similarity": 0.88, "source": "iNaturalist"}
    ])
    assert neighbours.iloc[0]["taxon"] == "Chironomus"
    assert neighbours.iloc[0]["similarity"] == "88.0%"


def test_open_set_message_is_conservative():
    text = accepted_markdown({
        "accepted": {"rank": "family", "taxon": "Chironomidae"},
        "open_set": {"rejected": True, "rank": "genus"},
    })
    assert "Unknown" in text
    assert "genus" in text


def test_key_guide_and_links_are_rendered():
    guide = {
        "title": "Adult Chironomidae evidence",
        "scope": "adult",
        "coverage": "family-specific",
        "candidate_genera": ["Chironomus", "Dicrotendipes"],
        "steps": [
            {"title": "Wing and squama", "instruction": "Check setae."},
            {"title": "Male hypopygium", "instruction": "Request a second view."},
        ],
        "stop_rules": ["Stop when the required structure is hidden."],
    }
    assert "Chironomus" in guide_markdown(guide)
    assert checklist_choices(guide) == ["Wing and squama", "Male hypopygium"]

    references = keys_markdown({
        "results": [{
            "title": "A key to Chironomidae",
            "url": "https://example.org/key",
            "provider": "curated",
        }]
    })
    assert "[A key to Chironomidae](https://example.org/key)" in references
