from pathlib import Path

from diptera_id.corpus.io import ManifestWriter, load_manifest
from diptera_id.corpus.schema import finalize_record, normalize_license


def test_license_normalization():
    assert normalize_license("https://creativecommons.org/licenses/by/4.0/") == "CC-BY"
    assert normalize_license("CC BY 3.0") == "CC-BY"
    assert normalize_license("cc-by-nc") == "CC-BY-NC"
    assert normalize_license("project-owned") == "PROJECT-OWNED"


def test_finalize_record_is_group_safe_and_compatible():
    row = finalize_record({
        "source": "local_verified",
        "source_record_id": "voucher-1",
        "source_image_id": "dorsal.jpg",
        "image_license": "project-owned",
        "order": "Diptera",
        "family": "Syrphidae",
        "species": "Eristalis tenax (Linnaeus, 1758)",
    })
    assert row["species"] == "Eristalis tenax"
    assert row["genus"] == "Eristalis"
    assert row["split_group"] == row["specimen_group_id"]
    assert row["photo_license"] == "PROJECT-OWNED"
    assert row["eligible_supervised"] is True


def test_csv_manifest_roundtrip(tmp_path: Path):
    out = tmp_path / "manifest.csv"
    with ManifestWriter(out) as writer:
        writer.write([finalize_record({
            "source": "test",
            "source_record_id": "1",
            "source_image_id": "a",
            "image_url": "https://example.org/a.jpg",
            "image_license": "CC0",
            "order": "Diptera",
            "family": "Muscidae",
        })])
    loaded = load_manifest(out)
    assert len(loaded) == 1
    assert loaded.iloc[0]["family"] == "Muscidae"


def test_subfamily_and_tribe_labels_are_supervised():
    from diptera_id.corpus.schema import finalize_record

    for rank, value in (
        ("subfamily", "Diapriinae"),
        ("tribe", "Diapriini"),
    ):
        record = finalize_record({
            "source": "BIOSCAN-5M",
            "source_record_id": f"specimen-{rank}",
            "source_image_id": f"image-{rank}",
            "local_path": f"/images/{rank}.jpg",
            "image_license": "CC-BY",
            "order": "Hymenoptera",
            "family": "Diapriidae",
            rank: value,
        })

        assert record["identification_level"] == rank
        assert record["eligible_supervised"] is True
        assert record["exclusion_reason"] == ""
