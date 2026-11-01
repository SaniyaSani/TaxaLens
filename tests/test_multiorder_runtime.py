from __future__ import annotations

from threading import RLock

import numpy as np
from PIL import Image

from diptera_id.multiorder_service import (
    MultiOrderTaxaLensRuntime,
)


class FakeRouter:
    classes_ = np.asarray(
        ["Diptera", "Hymenoptera"],
        dtype=object,
    )

    def __init__(self, probabilities):
        self.probabilities = np.asarray(
            probabilities,
            dtype=np.float32,
        )

    def predict_proba(self, values):
        assert values.shape == (1, 4)
        return self.probabilities.reshape(1, -1)


class FakeOrderRuntime:
    def __init__(self, order):
        self.order = order
        self._embedder = object()
        self.embed_calls = 0
        self.predict_calls = []

    def load(self):
        return self

    def embed(self, image):
        assert isinstance(image, Image.Image)
        self.embed_calls += 1
        return np.asarray(
            [0.1, 0.2, 0.3, 0.4],
            dtype=np.float32,
        )

    def predict_vector(
        self,
        vector,
        top_k=5,
        *,
        include_keys=True,
    ):
        self.predict_calls.append({
            "vector": np.asarray(vector),
            "top_k": top_k,
            "include_keys": include_keys,
        })
        return {
            "accepted": {
                "rank": "family",
                "taxon": f"{self.order}Family",
            },
            "predictions": {
                "family": [],
                "genus": [],
                "species": [],
            },
            "open_set": {"rejected": False},
            "similar_specimens": [],
            "diagnostics": {},
            "key_suggestions": {"results": []},
            "key_guide": {"steps": []},
            "warnings": [],
        }


def make_runtime(probabilities, threshold=0.8):
    runtime = MultiOrderTaxaLensRuntime.__new__(
        MultiOrderTaxaLensRuntime
    )
    runtime._lock = RLock()
    runtime._router = FakeRouter(probabilities)
    runtime._router_payload = {}
    runtime.acceptance_threshold = threshold
    runtime.encoder_config = {}
    runtime.diptera = FakeOrderRuntime("Diptera")
    runtime.hymenoptera = FakeOrderRuntime("Hymenoptera")
    return runtime


def specimen():
    return Image.new("RGB", (32, 32), "white")


def test_auto_routes_with_one_encoder_call():
    runtime = make_runtime([0.05, 0.95])

    payload = runtime.predict(specimen(), mode="Auto")

    assert payload["order"]["selected_order"] == "Hymenoptera"
    assert runtime.diptera.embed_calls == 1
    assert len(runtime.diptera.predict_calls) == 0
    assert len(runtime.hymenoptera.predict_calls) == 1
    assert (
        runtime.hymenoptera.predict_calls[0]["include_keys"]
        is False
    )
    assert (
        runtime.diptera._embedder
        is runtime.hymenoptera._embedder
    )


def test_manual_override_uses_requested_order():
    runtime = make_runtime([0.02, 0.98])

    payload = runtime.predict(specimen(), mode="Diptera")

    assert payload["order"]["predicted_order"] == "Hymenoptera"
    assert payload["order"]["selected_order"] == "Diptera"
    assert payload["order"]["manual_override"] is True
    assert runtime.diptera.embed_calls == 1
    assert len(runtime.diptera.predict_calls) == 1
    assert (
        runtime.diptera.predict_calls[0]["include_keys"]
        is True
    )


def test_uncertain_auto_route_stops_before_taxonomy():
    runtime = make_runtime(
        [0.45, 0.55],
        threshold=0.8,
    )

    payload = runtime.predict(specimen(), mode="Auto")

    assert payload["order"]["selected_order"] is None
    assert payload["open_set"]["rank"] == "order"
    assert runtime.diptera.embed_calls == 1
    assert not runtime.diptera.predict_calls
    assert not runtime.hymenoptera.predict_calls


def test_invalid_mode_is_rejected():
    runtime = make_runtime([0.9, 0.1])

    try:
        runtime.predict(specimen(), mode="Coleoptera")
    except ValueError as error:
        assert "mode must be one of" in str(error)
    else:
        raise AssertionError("Invalid mode was accepted")
