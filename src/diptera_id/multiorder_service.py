from __future__ import annotations

"""Unified Diptera and Hymenoptera inference runtime."""

import json
import os
from pathlib import Path
from threading import RLock
from typing import Any

import joblib
import numpy as np
from PIL import Image

from .demo_service import TaxaLensRuntime


ORDERS = ("Diptera", "Hymenoptera")
MODES = ("Auto", *ORDERS)


def _required_path(environment_name: str) -> Path:
    value = os.getenv(environment_name, "").strip()

    if not value:
        raise RuntimeError(
            f"Set {environment_name} before starting "
            "the multi-order runtime."
        )

    path = Path(value).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"{environment_name} does not exist: {path}"
        )

    return path


class MultiOrderTaxaLensRuntime:
    """One shared encoder with order-specific heads and retrieval."""

    def __init__(
        self,
        project_root: str | Path,
        diptera_model_dir: str | Path,
        hymenoptera_model_dir: str | Path,
        router_dir: str | Path,
    ):
        self.root = Path(project_root).expanduser().resolve()
        self.router_dir = Path(router_dir).expanduser().resolve()

        router_file = self.router_dir / "order_router.joblib"
        report_file = self.router_dir / "order_router_report.json"

        if not router_file.is_file():
            raise FileNotFoundError(router_file)

        if not report_file.is_file():
            raise FileNotFoundError(report_file)

        self.diptera = TaxaLensRuntime(
            self.root,
            diptera_model_dir,
        )
        self.hymenoptera = TaxaLensRuntime(
            self.root,
            hymenoptera_model_dir,
        )

        self._lock = RLock()
        self._router = None
        self._router_payload: dict[str, Any] = {}
        self.acceptance_threshold = 1.0
        self.encoder_config: dict[str, Any] = {}

    @classmethod
    def from_environment(
        cls,
        project_root: str | Path,
    ) -> "MultiOrderTaxaLensRuntime":
        return cls(
            project_root,
            _required_path("TAXALENS_DIPTERA_MODEL_DIR"),
            _required_path("TAXALENS_HYMENOPTERA_MODEL_DIR"),
            _required_path("TAXALENS_ORDER_ROUTER_DIR"),
        )

    def load(self) -> "MultiOrderTaxaLensRuntime":
        with self._lock:
            if self._router is None:
                payload = joblib.load(
                    self.router_dir / "order_router.joblib"
                )

                if not isinstance(payload, dict):
                    raise TypeError(
                        "Order-router payload must be a dictionary"
                    )

                model = payload.get("model")

                if model is None or not hasattr(
                    model,
                    "predict_proba",
                ):
                    raise TypeError(
                        "Order-router classifier is missing"
                    )

                classes = tuple(
                    str(value)
                    for value in getattr(
                        model,
                        "classes_",
                        (),
                    )
                )

                if set(classes) != set(ORDERS):
                    raise ValueError(
                        f"Unexpected router classes: {classes}"
                    )

                self._router_payload = payload
                self._router = model
                self.acceptance_threshold = float(
                    payload["acceptance_threshold"]
                )
                self.encoder_config = dict(
                    payload.get("encoder_config") or {}
                )

            self.diptera.load()

            # Inject the already-loaded fusion encoder before loading
            # the Hymenoptera heads. This prevents a second backbone.
            self.hymenoptera._embedder = (
                self.diptera._embedder
            )
            self.hymenoptera.load()

            if (
                self.diptera._embedder
                is not self.hymenoptera._embedder
            ):
                raise RuntimeError(
                    "Diptera and Hymenoptera do not share "
                    "one encoder instance"
                )

        return self

    @property
    def shared_encoder(self):
        self.load()
        return self.diptera._embedder

    def route_vector(
        self,
        vector,
    ) -> dict[str, Any]:
        self.load()

        values = np.asarray(
            vector,
            dtype=np.float32,
        ).reshape(1, -1)

        probabilities = self._router.predict_proba(values)[0]
        classes = [
            str(value)
            for value in self._router.classes_
        ]
        scores = {
            name: float(score)
            for name, score in zip(classes, probabilities)
        }
        predicted = max(scores, key=scores.get)
        confidence = scores[predicted]

        return {
            "predicted_order": predicted,
            "confidence": confidence,
            "threshold": self.acceptance_threshold,
            "router_accepted": (
                confidence >= self.acceptance_threshold
            ),
            "probabilities": scores,
        }

    def predict(
        self,
        image: Image.Image,
        *,
        mode: str = "Auto",
        top_k: int = 5,
    ) -> dict[str, Any]:
        if mode not in MODES:
            raise ValueError(
                f"mode must be one of {MODES}, received {mode!r}"
            )

        self.load()

        # The expensive DINOv3 + BioCLIP operation happens once.
        vector = self.diptera.embed(image)
        routing = self.route_vector(vector)

        if mode == "Auto":
            selected_order = (
                routing["predicted_order"]
                if routing["router_accepted"]
                else None
            )
            manual_override = False
        else:
            selected_order = mode
            manual_override = True

        routing.update({
            "mode": mode,
            "selected_order": selected_order,
            "manual_override": manual_override,
        })

        if selected_order is None:
            return {
                "order": routing,
                "accepted": {
                    "rank": None,
                    "taxon": None,
                },
                "predictions": {
                    "family": [],
                    "genus": [],
                    "species": [],
                },
                "open_set": {
                    "rejected": True,
                    "rank": "order",
                    "taxon": routing["predicted_order"],
                    "reason": (
                        "order_router_below_confidence_threshold"
                    ),
                },
                "similar_specimens": [],
                "diagnostics": {},
                "key_suggestions": {"results": []},
                "key_guide": {
                    "title": "Order uncertain",
                    "scope": "manual review",
                    "coverage": "Diptera or Hymenoptera",
                    "candidate_genera": [],
                    "steps": [],
                    "stop_rules": [
                        "Choose Diptera or Hymenoptera manually "
                        "only when morphology supports it."
                    ],
                },
                "warnings": [
                    (
                        "The order router was not confident enough "
                        "to select a taxonomic model."
                    )
                ],
            }

        runtime = (
            self.diptera
            if selected_order == "Diptera"
            else self.hymenoptera
        )

        payload = runtime.predict_vector(
            vector,
            top_k=top_k,
            include_keys=(selected_order == "Diptera"),
        )
        payload["order"] = routing
        return payload

    def configuration(self) -> dict[str, Any]:
        self.load()

        with (
            self.router_dir / "order_router_report.json"
        ).open(encoding="utf-8") as handle:
            report = json.load(handle)

        return {
            "orders": list(ORDERS),
            "modes": list(MODES),
            "acceptance_threshold": (
                self.acceptance_threshold
            ),
            "encoder_config": self.encoder_config,
            "router_report": report,
            "shared_encoder": True,
        }
