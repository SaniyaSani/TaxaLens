#!/usr/bin/env python3
from __future__ import annotations

"""Publish the trained bundle and the free TaxaLens ZeroGPU Space.

This is a one-time deployment helper.  The resulting public app does not need
Colab; Colab is only a convenient place from which to transfer artefacts that
already live on Google Drive.
"""

import argparse
import getpass
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diptera_id.demo_service import MODEL_DOWNLOAD_PATTERNS, validate_model_dir


def login_token() -> str:
    from huggingface_hub import get_token, login

    token = get_token()
    if not token:
        login(skip_if_logged_in=True)
        token = get_token()
    if not token:
        raise SystemExit("STOP: Hugging Face login did not produce a token")
    return token


def runtime_read_token() -> str:
    """Get a non-expiring read token without echoing it into notebook output."""

    token = os.getenv("HF_RUNTIME_TOKEN", "").strip()
    if token:
        return token
    print(
        "Create a READ token at https://huggingface.co/settings/tokens . "
        "It must belong to the same account that has DINOv3 access.\n"
        "Paste it into the hidden field below; it will be saved only as an encrypted Space secret.",
        flush=True,
    )
    token = getpass.getpass("HF read token for the running website: ").strip()
    if not token:
        raise SystemExit("STOP: a read token is required by the public Space")
    return token


def model_operations(model_dir: Path):
    from huggingface_hub import CommitOperationAdd

    operations = []
    for filename in MODEL_DOWNLOAD_PATTERNS:
        path = model_dir / filename
        if path.is_file():
            operations.append(CommitOperationAdd(path_in_repo=filename, path_or_fileobj=path))
    card = b"""---
library_name: scikit-learn
pipeline_tag: image-classification
private: true
---

# TaxaLens v0.9 inference artefacts

Private deployment bundle for the TaxaLens research prototype. It contains the
hierarchical classifiers and retrieval index, but not the training image corpus.
Species outputs are hypotheses requiring morphological verification.
"""
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card))
    return operations


def space_operations():
    from huggingface_hub import CommitOperationAdd

    files: list[tuple[str, Path]] = [
        ("README.md", ROOT / "SPACE_README.md"),
        ("app.py", ROOT / "app.py"),
        ("requirements.txt", ROOT / "requirements.txt"),
        ("requirements-corpus.txt", ROOT / "requirements-corpus.txt"),
        ("requirements-embeddings.txt", ROOT / "requirements-embeddings.txt"),
        ("pyproject.toml", ROOT / "pyproject.toml"),
    ]
    for path in sorted((ROOT / "src").rglob("*.py")):
        files.append((path.relative_to(ROOT).as_posix(), path))
    for path in sorted((ROOT / "data").glob("*.json")):
        files.append((path.relative_to(ROOT).as_posix(), path))
    return [
        CommitOperationAdd(path_in_repo=destination, path_or_fileobj=source)
        for destination, source in files
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the free TaxaLens ZeroGPU demo")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--space-name", default="TaxaLens")
    parser.add_argument("--model-repo-name", default="TaxaLens-v09-models")
    args = parser.parse_args()

    model_dir = validate_model_dir(args.model_dir)
    token = login_token()

    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import SpaceHardware

    api = HfApi(token=token)
    identity = api.whoami()
    owner = identity.get("name") or identity.get("fullname")
    if not owner:
        raise SystemExit("STOP: could not determine the Hugging Face account name")

    model_repo_id = f"{owner}/{args.model_repo_name}"
    space_id = f"{owner}/{args.space_name}"

    print(f"1/4 Creating private model repository: {model_repo_id}", flush=True)
    api.create_repo(
        repo_id=model_repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
    )
    print("2/4 Uploading inference artefacts (the large retrieval vectors can take time)…", flush=True)
    api.create_commit(
        repo_id=model_repo_id,
        repo_type="model",
        operations=model_operations(model_dir),
        commit_message="Upload TaxaLens v0.9 inference bundle",
    )

    runtime_token = runtime_read_token()

    print(f"3/4 Creating public ZeroGPU Space: {space_id}", flush=True)
    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            private=False,
            exist_ok=True,
            space_sdk="gradio",
            space_hardware=SpaceHardware.ZERO_A10G,
        )
    except Exception as exc:
        raise SystemExit(
            "STOP: Hugging Face did not allow creation of a ZeroGPU Space. "
            "A free account must be in good standing, have a verified email, and normally be "
            "older than 30 days. The model repository upload is safe and does not need repeating. "
            f"Details: {type(exc).__name__}: {exc}"
        ) from exc

    # The token is stored as an encrypted Space secret. It lets the running app
    # read the private TaxaLens bundle and the gated DINOv3 checkpoint.
    api.add_space_secret(
        repo_id=space_id,
        key="HF_TOKEN",
        value=runtime_token,
        description="Read the private TaxaLens bundle and gated DINOv3 checkpoint",
    )
    api.add_space_variable(
        repo_id=space_id,
        key="TAXALENS_MODEL_REPO",
        value=model_repo_id,
        description="Private inference bundle repository",
    )
    api.request_space_hardware(space_id, SpaceHardware.ZERO_A10G)

    print("4/4 Uploading the website…", flush=True)
    api.create_commit(
        repo_id=space_id,
        repo_type="space",
        operations=space_operations(),
        commit_message="Deploy TaxaLens ZeroGPU demo",
    )

    print("\n✓ PUBLIC TAXALENS DEPLOYMENT STARTED", flush=True)
    print(f"Space: https://huggingface.co/spaces/{space_id}", flush=True)
    print(f"Direct app: https://{owner}-{args.space_name}.hf.space", flush=True)
    print(f"Private model bundle: https://huggingface.co/{model_repo_id}", flush=True)
    print("Wait for the Space status to change from Building to Running.", flush=True)


if __name__ == "__main__":
    main()
