from __future__ import annotations

"""Public HTTP API for the unified TaxaLens runtime."""

import io
import os
import sys
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from diptera_id.multiorder_service import (  # noqa: E402
    MODES,
    MultiOrderTaxaLensRuntime,
)


MAX_IMAGE_BYTES = int(
    os.getenv("TAXALENS_MAX_IMAGE_BYTES", str(15 * 1024 * 1024))
)
_runtime: MultiOrderTaxaLensRuntime | None = None
_runtime_lock = Lock()


def allowed_origins() -> list[str]:
    raw = os.getenv(
        "TAXALENS_ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    )
    return [value.strip() for value in raw.split(",") if value.strip()]


def get_runtime() -> MultiOrderTaxaLensRuntime:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                runtime = MultiOrderTaxaLensRuntime.from_environment(ROOT)
                runtime.load()
                _runtime = runtime
    return _runtime


def hymenoptera_guidance(family: str, genera: list[str]) -> dict:
    resources = [
        {
            "title": "Hymenoptera of the world: an identification guide to families",
            "url": "https://www.publications.gc.ca/site/eng/9.644683/publication.html",
            "scope": "Worldwide family-level guide; verify current family concepts.",
        }
    ]
    parasitoid_families = {
        "Braconidae",
        "Ichneumonidae",
        "Figitidae",
        "Pteromalidae",
        "Eulophidae",
        "Mymaridae",
        "Platygastridae",
        "Diapriidae",
        "Encyrtidae",
    }
    if family in parasitoid_families:
        resources.append(
            {
                "title": "Interactive key to Central European parasitic wasps",
                "url": "https://parasitic-wasps.identificationkey.org/mkey.html",
                "scope": "Regional higher-level parasitoid key.",
            }
        )

    return {
        "title": "What to inspect next",
        "working_hypothesis": family or None,
        "candidate_genera": genera[:3],
        "steps": [
            {
                "title": "Head and antennae",
                "inspect": "Add sharp frontal and lateral close-ups.",
                "why": "Antenna and facial structures are used by many applicable keys.",
            },
            {
                "title": "Wing base and venation",
                "inspect": "Photograph a wing flat, including its base, veins and margin.",
                "why": "Venation frequently separates higher taxa and genera.",
            },
            {
                "title": "Body junction, propodeum and hind legs",
                "inspect": "Capture these structures from more than one angle.",
                "why": "A single habitus image often hides decisive structures.",
            },
            {
                "title": "Sex and life stage",
                "inspect": "Record them before using a sex- or stage-specific key.",
                "why": "Some key paths are not interchangeable.",
            },
        ],
        "resources": resources,
        "disclaimer": (
            "These are prioritized views to acquire, not image characters "
            "verified by the current classifier."
        ),
    }


def enrich_payload(payload: dict) -> dict:
    order = payload.get("order") or {}
    selected = order.get("selected_order")
    family_rows = (payload.get("predictions") or {}).get("family") or []
    genus_rows = (payload.get("predictions") or {}).get("genus") or []
    family = str(family_rows[0].get("taxon") or "") if family_rows else ""
    genera = [
        str(row.get("taxon"))
        for row in genus_rows[:4]
        if row.get("taxon")
    ]

    if selected == "Hymenoptera":
        payload["guided_verification"] = hymenoptera_guidance(
            family,
            genera,
        )
    else:
        guide = payload.get("key_guide") or {}
        payload["guided_verification"] = {
            **guide,
            "resources": (
                payload.get("key_suggestions") or {}
            ).get("results", []),
            "disclaimer": (
                guide.get("disclaimer")
                or "Key leads require independent morphological verification."
            ),
        }
    return payload


app = FastAPI(
    title="TaxaLens API",
    version="1.0.0",
    description="Diptera and Hymenoptera image identification.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _runtime is not None,
        "orders": ["Diptera", "Hymenoptera"],
    }


@app.post("/v1/identify")
async def identify(
    image: UploadFile = File(...),
    mode: str = Form("Auto"),
) -> dict:
    if mode not in MODES:
        raise HTTPException(
            status_code=422,
            detail=f"mode must be one of {MODES}",
        )
    data = await image.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large")

    try:
        with Image.open(io.BytesIO(data)) as opened:
            specimen = opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=415,
            detail="The uploaded file is not a readable image",
        ) from exc

    try:
        runtime = await run_in_threadpool(get_runtime)
        payload = await run_in_threadpool(
            lambda: runtime.predict(specimen, mode=mode)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exc).__name__}",
        ) from exc

    return enrich_payload(payload)


@app.get("/v1/configuration")
def configuration() -> dict:
    return get_runtime().configuration()

