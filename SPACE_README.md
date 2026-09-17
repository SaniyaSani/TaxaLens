---
title: TaxaLens
emoji: 🪰
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
short_description: Evidence-first identification of small Diptera with DINOv3 and BioCLIP
models:
  - facebook/dinov3-vitb16-pretrain-lvd1689m
  - imageomics/bioclip-2
---

# TaxaLens public research demo

Upload a Diptera photograph to obtain conservative family/genus/species candidates,
nearest reference specimens, and a source-backed morphology checklist. Species output
is a hypothesis, not a determination.

The trained TaxaLens artefacts are stored separately in a private model repository.
The Space reads its repository name from `TAXALENS_MODEL_REPO` and uses the private
`HF_TOKEN` Space secret both for that repository and the gated DINOv3 backbone.
