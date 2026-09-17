from __future__ import annotations

"""Shared inference service for the browser demos.

The training pipeline deliberately keeps model artefacts outside the source
repository.  A deployed demo can either receive a local model directory or
download the small inference bundle from a private Hugging Face model repo.
"""

import os
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

from PIL import Image, ImageOps

from .embedding import embedder_from_config
from .hierarchy import load_classifier_bundle
from .key_finder import KeyFinder, candidate_genera
from .key_navigator import KeyNavigator
from .morphology import diagnostic_help
from .retrieval import RetrievalIndex


REQUIRED_MODEL_FILES = (
    "classifiers.joblib",
    "retrieval_vectors.npy",
    "retrieval_metadata.csv",
)

MODEL_DOWNLOAD_PATTERNS = [
    *REQUIRED_MODEL_FILES,
    "retrieval.index",
    "embedding_config.json",
    "training_report_hierarchical.json",
    "evaluation_by_source.md",
    "encoder_comparison.json",
    "encoder_comparison.md",
]


def validate_model_dir(path: str | Path) -> Path:
    folder = Path(path).expanduser().resolve()
    missing = [name for name in REQUIRED_MODEL_FILES if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete TaxaLens model directory {folder}; missing: {', '.join(missing)}"
        )
    return folder


def resolve_model_dir(cache_dir: str | Path = "/tmp/taxalens-model") -> Path:
    """Resolve local artefacts or download them from a private HF model repo."""

    local = os.getenv("TAXALENS_MODEL_DIR", "").strip()
    if local:
        return validate_model_dir(local)

    repo_id = os.getenv("TAXALENS_MODEL_REPO", "").strip()
    if not repo_id:
        raise RuntimeError(
            "Set TAXALENS_MODEL_DIR or TAXALENS_MODEL_REPO before starting the demo."
        )

    from huggingface_hub import snapshot_download

    downloaded = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=os.getenv("HF_TOKEN") or None,
        allow_patterns=MODEL_DOWNLOAD_PATTERNS,
        local_dir=str(Path(cache_dir).expanduser()),
    )
    return validate_model_dir(downloaded)


class TaxaLensRuntime:
    """Thread-safe lazy loader and evidence-first prediction facade."""

    def __init__(self, project_root: str | Path, model_dir: str | Path):
        self.root = Path(project_root).expanduser().resolve()
        self.model_dir = validate_model_dir(model_dir)
        self._lock = RLock()
        self._embedder = None
        self._classifiers = None
        self._retrieval = None
        cache_dir = Path(os.getenv("TAXALENS_KEY_CACHE", "/tmp/taxalens-key-cache"))
        self.key_finder = KeyFinder(self.root / "data" / "key_catalog_v09.json", cache_dir)
        self.key_navigator = KeyNavigator(self.root / "data" / "key_navigator_v09.json")

    def load(self) -> "TaxaLensRuntime":
        with self._lock:
            if self._classifiers is None:
                self._classifiers = load_classifier_bundle(
                    self.model_dir / "classifiers.joblib"
                )
            if self._embedder is None:
                embedding = self._classifiers.metadata.get("embedding", {})
                self._embedder = embedder_from_config(embedding)
            if self._retrieval is None:
                self._retrieval = RetrievalIndex.load(self.model_dir)
        return self

    def predict(self, image: Image.Image, top_k: int = 5) -> dict[str, Any]:
        if image is None:
            raise ValueError("Upload an image first")
        specimen = ImageOps.exif_transpose(image).convert("RGB")
        with self._lock:
            self.load()
            embedding_config = self._classifiers.metadata.get("embedding", {})
            tile_grid = int(embedding_config.get("tile_grid", 1))
            include_whole = bool(embedding_config.get("include_whole", True))
            vector = (
                self._embedder.embed_multicrop(
                    specimen,
                    tile_grid=tile_grid,
                    include_whole=include_whole,
                )
                if tile_grid > 1
                else self._embedder.embed_one(specimen)
            )
            predictions = self._classifiers.predict_all(vector, top_k=top_k)
            neighbours = self._retrieval.search(vector, k=8)
            open_set = getattr(
                self._classifiers,
                "last_open_set",
                {"rejected": False},
            )

        best_family = (
            predictions.get("family", [{}])[0].get("taxon")
            if predictions.get("family")
            else None
        )
        genera = candidate_genera(predictions.get("genus", []))
        thresholds = {"family": 0.40, "genus": 0.50, "species": 0.65}
        accepted_rank = None
        accepted_taxon = None
        for rank in ("family", "genus", "species"):
            candidates = predictions.get(rank) or []
            if open_set.get("rejected") and open_set.get("rank") == rank:
                break
            if candidates and candidates[0]["probability"] >= thresholds[rank]:
                accepted_rank = rank
                accepted_taxon = candidates[0]["taxon"]
            else:
                break

        return {
            "accepted": {"rank": accepted_rank, "taxon": accepted_taxon},
            "predictions": predictions,
            "open_set": open_set,
            "similar_specimens": neighbours,
            "diagnostics": diagnostic_help(
                best_family,
                self.root / "data" / "morphology_rules.json",
                self.root / "data" / "key_references.json",
            ),
            "key_suggestions": self.key_finder.find(
                best_family,
                genera,
                live=False,
            ),
            "key_guide": self.key_navigator.build(best_family, genera),
            "warnings": [
                "Probabilities are model scores, not taxonomic certainty.",
                "A species suggestion is a candidate, not a determination; verify it in an applicable key.",
                "Field photos and preserved specimens are different visual domains.",
                "If a required structure is invisible, stop at family/genus or request another view, preparation, DNA, or specialist review.",
            ],
        }

    def guide(self, family: str | None, genera: Iterable[str]) -> dict[str, Any]:
        return self.key_navigator.build(family, genera)

    def keys(
        self,
        family: str | None,
        genera: Iterable[str],
        *,
        live: bool = False,
    ) -> dict[str, Any]:
        return self.key_finder.find(family, genera, live=live)
