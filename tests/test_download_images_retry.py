from __future__ import annotations

from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

from scripts import download_images


class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"") -> None:
        self.status_code = status_code
        self.content = content
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def close(self) -> None:
        self.closed = True

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = requests.Response()
            response.status_code = self.status_code
            response.url = "https://example.test/image.jpg"
            raise requests.HTTPError(
                f"HTTP {self.status_code}",
                response=response,
            )


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = iter(responses)
        self.calls: list[tuple[str, object]] = []

    def get(self, url: str, timeout):
        self.calls.append((url, timeout))
        return next(self.responses)


def jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(120, 80, 40)).save(
        buffer,
        format="JPEG",
    )
    return buffer.getvalue()


def raw_record(suffix: str) -> dict[str, str]:
    return {
        "source": "GBIF",
        "source_record_id": f"specimen-{suffix}",
        "source_image_id": f"image-{suffix}",
        "image_url": f"https://example.test/{suffix}.jpg",
        "image_license": "CC-BY",
        "order": "Hymenoptera",
        "family": "Ichneumonidae",
        "species": "Rhyssa persuasoria",
    }


def test_transient_http_error_is_retried(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session = FakeSession([
        FakeResponse(503),
        FakeResponse(200, jpeg_bytes()),
    ])

    monkeypatch.setattr(
        download_images,
        "session_for_thread",
        lambda: session,
    )
    monkeypatch.setattr(download_images.time, "sleep", lambda _: None)

    record, attempted, downloaded = download_images.cache_record(
        raw_record("retry"),
        {"CC-BY"},
        tmp_path,
        0,
        connect_timeout=3,
        read_timeout=9,
        retries=1,
        retry_backoff=0,
    )

    assert attempted == 1
    assert downloaded == 1
    assert len(session.calls) == 2
    assert all(timeout == (3, 9) for _, timeout in session.calls)
    assert Path(record["local_path"]).is_file()
    assert record["image_sha256"]


def test_permanent_http_error_is_not_retried(
    tmp_path: Path,
    monkeypatch,
) -> None:
    session = FakeSession([
        FakeResponse(404),
        FakeResponse(200, jpeg_bytes()),
    ])

    monkeypatch.setattr(
        download_images,
        "session_for_thread",
        lambda: session,
    )
    monkeypatch.setattr(download_images.time, "sleep", lambda _: None)

    record, attempted, downloaded = download_images.cache_record(
        raw_record("missing"),
        {"CC-BY"},
        tmp_path,
        0,
        connect_timeout=3,
        read_timeout=9,
        retries=3,
        retry_backoff=0,
    )

    assert attempted == 1
    assert downloaded == 0
    assert len(session.calls) == 1
    assert record["local_path"] == ""
    assert record["exclusion_reason"] == "download_error:HTTPError"
