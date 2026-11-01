from __future__ import annotations

"""Public multi-order TaxaLens demo."""

import os
import sys
from pathlib import Path

import gradio as gr

try:
    import spaces
except ImportError:
    class _Spaces:
        @staticmethod
        def GPU(*decorator_args, **decorator_kwargs):
            if decorator_args and callable(decorator_args[0]):
                return decorator_args[0]

            def decorate(function):
                return function

            return decorate

    spaces = _Spaces()


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from diptera_id.gradio_ui import (
    accepted_markdown,
    checklist_choices,
    guide_markdown,
    keys_markdown,
    neighbour_frame,
    neighbour_gallery,
    prediction_frame,
)
from diptera_id.multiorder_service import MultiOrderTaxaLensRuntime


RUNTIME = None
STARTUP_ERROR = None

try:
    RUNTIME = MultiOrderTaxaLensRuntime.from_environment(ROOT)

    if os.getenv("SPACE_ID"):
        RUNTIME.load()
except Exception as exc:
    STARTUP_ERROR = f"{type(exc).__name__}: {exc}"


def _require_runtime() -> MultiOrderTaxaLensRuntime:
    if RUNTIME is None:
        raise gr.Error(
            "The multi-order model bundles are not configured. Set "
            "TAXALENS_DIPTERA_MODEL_DIR, "
            "TAXALENS_HYMENOPTERA_MODEL_DIR and "
            "TAXALENS_ORDER_ROUTER_DIR. "
            f"Startup detail: {STARTUP_ERROR}"
        )

    return RUNTIME


def _selected_order(payload: dict) -> str:
    order = payload.get("order") or {}
    return str(
        order.get("selected")
        or order.get("selected_order")
        or ""
    )


def _order_markdown(payload: dict) -> str:
    order = payload.get("order") or {}
    selected = _selected_order(payload)
    suggested = str(
        order.get("suggested")
        or order.get("predicted")
        or order.get("router_order")
        or order.get("routed_order")
        or order.get("candidate")
        or order.get("label")
        or selected
        or "Unknown"
    )

    try:
        confidence = float(order.get("confidence"))
        confidence_text = f"{confidence:.1%}"
    except (TypeError, ValueError):
        confidence_text = "—"

    try:
        threshold = float(order.get("threshold"))
        threshold_text = f"{threshold:.1%}"
    except (TypeError, ValueError):
        threshold_text = "—"

    if not selected:
        return (
            "### Order uncertain\n"
            f"Best router candidate: **{suggested}** "
            f"({confidence_text}); automatic threshold: "
            f"{threshold_text}.\n\n"
            "Choose Diptera or Hymenoptera manually, or provide "
            "another view."
        )

    mode = str(order.get("mode") or "Auto")
    return (
        f"### Selected order: **{selected}**\n"
        f"Router confidence: **{confidence_text}** · "
        f"mode: **{mode}**"
    )


def _diptera_key_outputs(
    family: str | None,
    genera: list[str],
    *,
    live: bool = False,
):
    runtime = _require_runtime()
    guide = runtime.diptera.guide(family, genera)
    keys = runtime.diptera.keys(family, genera, live=live)

    return (
        guide_markdown(guide),
        gr.CheckboxGroup(
            choices=checklist_choices(guide),
            value=[],
        ),
        keys_markdown(keys),
    )


@spaces.GPU(duration=180)
def identify(image, order_mode):
    runtime = _require_runtime()

    if image is None:
        raise gr.Error("Upload a specimen image first.")

    payload = runtime.predict(image, mode=order_mode)
    selected_order = _selected_order(payload)

    predictions = payload.get("predictions") or {}
    neighbours = payload.get("similar_specimens") or []

    family_rows = predictions.get("family") or []
    genus_rows = predictions.get("genus") or []
    species_rows = predictions.get("species") or []

    family = (
        family_rows[0].get("taxon")
        if family_rows
        else ""
    )
    genera = [
        row.get("taxon")
        for row in genus_rows[:4]
        if row.get("taxon")
    ]
    selected_genus = genera[0] if genera else None

    use_diptera_keys = selected_order == "Diptera"

    if use_diptera_keys:
        guide = payload.get("key_guide") or {}
        keys = payload.get("key_suggestions") or {}

        if not guide:
            guide = runtime.diptera.guide(family, genera)
        if not keys:
            keys = runtime.diptera.keys(
                family,
                genera,
                live=False,
            )

        guide_output = guide_markdown(guide)
        checks_output = gr.CheckboxGroup(
            choices=checklist_choices(guide),
            value=[],
        )
        keys_output = keys_markdown(keys)
    elif selected_order == "Hymenoptera":
        guide_output = (
            "### Hymenoptera evidence\n"
            "The Hymenoptera classifier and similar-specimen "
            "retrieval are active. A dedicated Hymenoptera "
            "morphology navigator is not attached yet."
        )
        checks_output = gr.CheckboxGroup(
            choices=[],
            value=[],
        )
        keys_output = (
            "Diptera keys are intentionally disabled for "
            "Hymenoptera results."
        )
    else:
        guide_output = (
            "### Order review required\n"
            "Select an order manually before using "
            "order-specific morphology tools."
        )
        checks_output = gr.CheckboxGroup(
            choices=[],
            value=[],
        )
        keys_output = "No order-specific key selected."

    warnings = "\n".join(
        f"- {item}"
        for item in payload.get("warnings") or []
    )

    return (
        _order_markdown(payload),
        accepted_markdown(payload),
        prediction_frame(family_rows),
        prediction_frame(genus_rows),
        prediction_frame(species_rows),
        neighbour_frame(neighbours),
        neighbour_gallery(neighbours),
        gr.Dropdown(
            choices=genera if use_diptera_keys else [],
            value=(
                selected_genus
                if use_diptera_keys
                else None
            ),
            interactive=use_diptera_keys,
        ),
        family,
        selected_order,
        guide_output,
        checks_output,
        keys_output,
        gr.Button(interactive=use_diptera_keys),
        warnings,
    )


def select_genus(
    family: str,
    genus: str,
    selected_order: str,
):
    if selected_order != "Diptera":
        return (
            "Diptera morphology tools are disabled for this order.",
            gr.CheckboxGroup(choices=[], value=[]),
            "No Diptera key search was performed.",
        )

    genera = [genus] if genus else []
    return _diptera_key_outputs(
        family,
        genera,
        live=False,
    )


def search_live_keys(
    family: str,
    genus: str,
    selected_order: str,
):
    if selected_order != "Diptera":
        return "Live Diptera key search is disabled for this order."

    genera = [genus] if genus else []
    runtime = _require_runtime()

    return keys_markdown(
        runtime.diptera.keys(
            family,
            genera,
            live=True,
        )
    )


CSS = """
:root {
  --taxa-ink:#17221c;
  --taxa-green:#2f644b;
  --taxa-cream:#f7f3e8;
}
.gradio-container {
  max-width:1280px !important;
  background:var(--taxa-cream);
}
#taxalens-hero {
  border:1px solid #17221c22;
  border-radius:24px;
  padding:26px;
  background:linear-gradient(135deg,#fdfbf5,#e4eee4);
  margin-bottom:16px;
}
#taxalens-hero h1 {
  color:var(--taxa-ink);
  letter-spacing:-0.04em;
}
.primary {
  background:var(--taxa-green) !important;
  border-color:var(--taxa-green) !important;
}
.taxa-note {
  font-size:.92rem;
  color:#45544a;
}
"""


with gr.Blocks(
    title="TaxaLens · Diptera + Hymenoptera",
    css=CSS,
) as demo:
    family_state = gr.State("")
    order_state = gr.State("")

    gr.Markdown(
        """
        # TaxaLens
        **Evidence-first identification of Diptera and Hymenoptera**

        One shared DINOv3 + BioCLIP encoder · automatic order router ·
        separate hierarchical taxonomic models · similar reference specimens
        """,
        elem_id="taxalens-hero",
    )

    if STARTUP_ERROR:
        gr.Markdown(
            "⚠️ **Deployment setup is incomplete.** "
            "The owner still needs to attach the model bundles."
        )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            image = gr.Image(
                type="pil",
                label="Specimen photograph",
                sources=["upload", "webcam", "clipboard"],
                height=470,
            )
            order_mode = gr.Radio(
                choices=["Auto", "Diptera", "Hymenoptera"],
                value="Auto",
                label="Order selection",
                info=(
                    "Auto uses the trained order router. "
                    "Manual selection overrides routing."
                ),
            )
            identify_button = gr.Button(
                "Identify specimen",
                variant="primary",
                size="lg",
            )
            gr.Markdown(
                "The first request after a sleeping Space may "
                "wait in the free GPU queue.",
                elem_classes=["taxa-note"],
            )

        with gr.Column(scale=7):
            order_result = gr.Markdown(
                "### Waiting for a specimen"
            )
            accepted = gr.Markdown(
                "No taxonomic prediction yet."
            )

            with gr.Tabs():
                with gr.Tab("Family"):
                    family_table = gr.Dataframe(
                        interactive=False
                    )
                with gr.Tab("Genus"):
                    genus_table = gr.Dataframe(
                        interactive=False
                    )
                with gr.Tab("Species candidate"):
                    species_table = gr.Dataframe(
                        interactive=False
                    )

    with gr.Tabs():
        with gr.Tab("Similar specimens"):
            neighbours_gallery = gr.Gallery(
                label="Nearest reference images available by URL",
                columns=4,
                object_fit="contain",
                height="auto",
            )
            neighbours_table = gr.Dataframe(
                interactive=False
            )

        with gr.Tab("Morphology & keys"):
            genus_choice = gr.Dropdown(
                label="Choose a genus candidate to inspect",
                choices=[],
                interactive=False,
            )
            key_guide = gr.Markdown(
                "Upload an image to activate applicable "
                "order-specific evidence tools."
            )
            evidence_checks = gr.CheckboxGroup(
                label="Visible evidence recorded",
                choices=[],
            )
            live_search_button = gr.Button(
                "Search current published Diptera keys",
                interactive=False,
            )
            key_results = gr.Markdown(
                "No key search has been performed."
            )

        with gr.Tab("Cautions"):
            warnings = gr.Markdown(
                "Species suggestions are hypotheses and require "
                "independent taxonomic verification."
            )

    identify_button.click(
        fn=identify,
        inputs=[image, order_mode],
        outputs=[
            order_result,
            accepted,
            family_table,
            genus_table,
            species_table,
            neighbours_table,
            neighbours_gallery,
            genus_choice,
            family_state,
            order_state,
            key_guide,
            evidence_checks,
            key_results,
            live_search_button,
            warnings,
        ],
    )

    genus_choice.change(
        fn=select_genus,
        inputs=[
            family_state,
            genus_choice,
            order_state,
        ],
        outputs=[
            key_guide,
            evidence_checks,
            key_results,
        ],
    )

    live_search_button.click(
        fn=search_live_keys,
        inputs=[
            family_state,
            genus_choice,
            order_state,
        ],
        outputs=[key_results],
    )


if __name__ == "__main__":
    share = (
        os.getenv("GRADIO_SHARE", "")
        .strip()
        .lower()
        in {"1", "true", "yes"}
    )

    demo.queue(
        default_concurrency_limit=1,
        max_size=12,
    ).launch(share=share)
