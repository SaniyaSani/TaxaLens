#!/usr/bin/env python3
"""Pre-cache every model required by one scalable TaxaLens profile.

Run this once before a Slurm embedding array.  It prevents many workers from
trying to download the same Hugging Face or BioCLIP files concurrently.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def profile_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def cache_dino(model_name: str) -> None:
    from transformers import AutoImageProcessor, AutoModel

    print(f"Caching DINO processor: {model_name}", flush=True)
    AutoImageProcessor.from_pretrained(model_name)
    print(f"Caching DINO model: {model_name}", flush=True)
    model = AutoModel.from_pretrained(model_name)
    print(f"DINO parameters: {sum(parameter.numel() for parameter in model.parameters()):,}", flush=True)
    del model
    gc.collect()


def cache_bioclip(model_name: str) -> None:
    import open_clip

    print(f"Caching BioCLIP model: {model_name}", flush=True)
    model, _, _ = open_clip.create_model_and_transforms(model_name)
    print(f"BioCLIP parameters: {sum(parameter.numel() for parameter in model.parameters()):,}", flush=True)
    del model
    gc.collect()


def cache_quality_model(model_name: str) -> None:
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    print(f"Caching quality processor: {model_name}", flush=True)
    Owlv2Processor.from_pretrained(model_name)
    print(f"Caching quality model: {model_name}", flush=True)
    model = Owlv2ForObjectDetection.from_pretrained(model_name)
    print(f"Quality-model parameters: {sum(parameter.numel() for parameter in model.parameters()):,}", flush=True)
    del model
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    path = profile_path(args.config)
    profile = json.loads(path.read_text(encoding="utf-8"))
    quality_model = str(
        profile.get("image_quality", {}).get(
            "model", "google/owlv2-base-patch16-ensemble"
        )
    )
    cache_quality_model(quality_model)

    encoders = profile.get("embedding", {}).get("encoders", [])
    if not encoders:
        encoders = [{"backend": "dino", **profile["embedding"]}]
    seen: set[tuple[str, str]] = set()
    for encoder in encoders:
        backend = str(encoder.get("backend", encoder.get("name", "dino"))).lower()
        model_name = str(encoder["model"])
        key = (backend, model_name)
        if key in seen:
            continue
        seen.add(key)
        if backend == "dino":
            cache_dino(model_name)
        elif backend == "bioclip":
            cache_bioclip(model_name)
        else:
            raise SystemExit(f"unsupported embedding backend: {backend}")
    print("All profile models are cached", flush=True)


if __name__ == "__main__":
    main()
