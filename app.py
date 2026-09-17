from __future__ import annotations

"""Public TaxaLens demo for a free Hugging Face ZeroGPU Space."""

import os
import sys
from pathlib import Path

import gradio as gr

try:
    import spaces
except ImportError:  # Local CPU preview and unit tests.
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

from diptera_id.demo_service import TaxaLensRuntime, resolve_model_dir
from diptera_id.gradio_ui import (
    accepted_markdown,
    checklist_choices,
    guide_markdown,
    keys_markdown,
    neighbour_frame,
    neighbour_gallery,
    prediction_frame,
)


RUNTIME = None
STARTUP_ERROR = None
try:
    RUNTIME = TaxaLensRuntime(ROOT, resolve_model_dir())
    # ZeroGPU emulates CUDA during startup. Loading here lets the platform move
    # both frozen encoders efficiently when the decorated function is called.
    if os.getenv("SPACE_ID"):
        RUNTIME.load()
except Exception as exc:  # Keep the UI alive with an actionable setup message.
    STARTUP_ERROR = f"{type(exc).__name__}: {exc}"


def _require_runtime() -> TaxaLensRuntime:
    if RUNTIME is None:
        raise gr.Error(
            "The model bundle is not configured yet. Space owner: add "
            "TAXALENS_MODEL_REPO and the HF_TOKEN secret. "
            f"Startup detail: {STARTUP_ERROR}"
        )
    return RUNTIME


def _key_outputs(family: str | None, genera: list[str], *, live: bool = False):
    runtime = _require_runtime()
    guide = runtime.guide(family, genera)
    keys = runtime.keys(family, genera, live=live)
    return (
        guide_markdown(guide),
        gr.CheckboxGroup(choices=checklist_choices(guide), value=[]),
        keys_markdown(keys),
    )


@spaces.GPU(duration=180)
def identify(image):
    runtime = _require_runtime()
    if image is None:
        raise gr.Error("Upload a specimen image first.")
    payload = runtime.predict(image)
    predictions = payload.get("predictions") or {}
    neighbours = payload.get("similar_specimens") or []
    family_rows = predictions.get("family") or []
    genus_rows = predictions.get("genus") or []
    species_rows = predictions.get("species") or []
    family = family_rows[0].get("taxon") if family_rows else ""
    genera = [row.get("taxon") for row in genus_rows[:4] if row.get("taxon")]
    selected_genus = genera[0] if genera else None
    guide = payload.get("key_guide") or runtime.guide(family, genera)
    keys = payload.get("key_suggestions") or runtime.keys(family, genera, live=False)
    warnings = "\n".join(f"- {item}" for item in payload.get("warnings") or [])
    return (
        accepted_markdown(payload),
        prediction_frame(family_rows),
        prediction_frame(genus_rows),
        prediction_frame(species_rows),
        neighbour_frame(neighbours),
        neighbour_gallery(neighbours),
        gr.Dropdown(choices=genera, value=selected_genus),
        family,
        guide_markdown(guide),
        gr.CheckboxGroup(choices=checklist_choices(guide), value=[]),
        keys_markdown(keys),
        warnings,
    )


def select_genus(family: str, genus: str):
    genera = [genus] if genus else []
    return _key_outputs(family, genera, live=False)


def search_live_keys(family: str, genus: str):
    genera = [genus] if genus else []
    runtime = _require_runtime()
    return keys_markdown(runtime.keys(family, genera, live=True))


CSS = """
:root { --taxa-ink:#17221c; --taxa-green:#2f644b; --taxa-cream:#f7f3e8; }
.gradio-container { max-width: 1280px !important; background: var(--taxa-cream); }
#taxalens-hero { border: 1px solid #17221c22; border-radius: 24px; padding: 26px;
  background: linear-gradient(135deg, #fdfbf5, #e4eee4); margin-bottom: 16px; }
#taxalens-hero h1 { color: var(--taxa-ink); letter-spacing: -0.04em; }
.primary { background: var(--taxa-green) !important; border-color: var(--taxa-green) !important; }
.taxa-note { font-size: .92rem; color: #45544a; }
"""


with gr.Blocks(title="TaxaLens · Evidence-first Diptera ID", css=CSS) as demo:
    family_state = gr.State("")
    gr.Markdown(
        """
        # TaxaLens
        **Evidence-first identification of small Diptera**

        DINOv3 + BioCLIP · hierarchical candidates · similar reference specimens · morphology navigator
        """,
        elem_id="taxalens-hero",
    )
    if STARTUP_ERROR:
        gr.Markdown(
            "⚠️ **Deployment setup is incomplete.** The owner still needs to attach the private model bundle."
        )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            image = gr.Image(
                type="pil",
                label="Specimen photograph",
                sources=["upload", "webcam", "clipboard"],
                height=470,
            )
            identify_button = gr.Button("Identify specimen", variant="primary", size="lg")
            gr.Markdown(
                "The first request after a sleeping Space may wait in the free GPU queue.",
                elem_classes=["taxa-note"],
            )
        with gr.Column(scale=7):
            accepted = gr.Markdown("### Waiting for a specimen")
            with gr.Tabs():
                with gr.Tab("Family"):
                    family_table = gr.Dataframe(interactive=False)
                with gr.Tab("Genus"):
                    genus_table = gr.Dataframe(interactive=False)
                with gr.Tab("Species candidate"):
                    species_table = gr.Dataframe(interactive=False)

    with gr.Tabs():
        with gr.Tab("Similar specimens"):
            neighbours_gallery = gr.Gallery(
                label="Nearest reference images available by URL",
                columns=4,
                object_fit="contain",
                height="auto",
            )
            neighbours_table = gr.Dataframe(interactive=False)
        with gr.Tab("Morphology & keys"):
            genus_choice = gr.Dropdown(
                label="Choose a genus candidate to inspect",
                choices=[],
                interactive=True,
            )
            key_guide = gr.Markdown(
                "Upload an image; then choose any genus candidate to rebuild the checklist."
            )
            evidence_checks = gr.CheckboxGroup(
                label="Visible evidence recorded",
                choices=[],
            )
            live_search_button = gr.Button("Search current published keys")
            key_results = gr.Markdown("Applicable references will appear here.")
        with gr.Tab("Cautions"):
            warnings = gr.Markdown(
                "Species suggestions are hypotheses and require morphological verification."
            )

    identify_event = identify_button.click(
        fn=identify,
        inputs=[image],
        outputs=[
            accepted,
            family_table,
            genus_table,
            species_table,
            neighbours_table,
            neighbours_gallery,
            genus_choice,
            family_state,
            key_guide,
            evidence_checks,
            key_results,
            warnings,
        ],
    )
    genus_choice.change(
        fn=select_genus,
        inputs=[family_state, genus_choice],
        outputs=[key_guide, evidence_checks, key_results],
    )
    live_search_button.click(
        fn=search_live_keys,
        inputs=[family_state, genus_choice],
        outputs=[key_results],
    )


if __name__ == "__main__":
    share = os.getenv("GRADIO_SHARE", "").strip().lower() in {"1", "true", "yes"}
    demo.queue(default_concurrency_limit=1, max_size=12).launch(share=share)
