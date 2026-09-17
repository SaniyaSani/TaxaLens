from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageOps
from .device import best_device


def patch_size_hint(model_name: str) -> int | None:
    """Best-effort patch-size hint without downloading the model config."""
    name = model_name.lower()
    if "dinov3" in name and ("vit" in name or "vits" in name or "vitb" in name or "vitl" in name or "vith" in name):
        return 16
    if "dinov2" in name:
        return 14
    return None


def validate_image_size(model_name: str, image_size: int) -> None:
    if image_size < 64:
        raise ValueError("image_size is implausibly small")
    patch = patch_size_hint(model_name)
    if patch and image_size % patch:
        raise ValueError(f"{model_name} expects an image size divisible by patch size {patch}; got {image_size}")


@dataclass
class DINOEmbedder:
    """Generic DINOv2/DINOv3 feature extractor.

    The project name is kept for backwards compatibility, but v0.8 defaults to
    DINOv3. The backbone stays frozen: downstream taxonomic heads are trained on
    normalized specimen embeddings.
    """

    model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    device: str | None = None
    image_size: int = 512

    def __post_init__(self) -> None:
        validate_image_size(self.model_name, self.image_size)
        self.device = self.device or best_device()
        from transformers import AutoImageProcessor, AutoModel
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval().to(self.device)
        config_patch = getattr(self.model.config, "patch_size", None)
        if isinstance(config_patch, (list, tuple)) and config_patch:
            config_patch = config_patch[0]
        if isinstance(config_patch, int) and self.image_size % config_patch:
            raise ValueError(
                f"image_size={self.image_size} is not divisible by loaded model patch_size={config_patch}"
            )

    def embed_images(self, images: Iterable[Image.Image]) -> np.ndarray:
        import torch

        images = [img.convert("RGB") for img in images]
        if not images:
            dimension = int(getattr(self.model.config, "hidden_size", 384))
            return np.empty((0, dimension), dtype=np.float32)

        # We explicitly pad to a square so whole-image geometry is preserved
        # instead of silently center-cropping away appendages or wing tips.
        padded = [
            ImageOps.pad(
                image,
                (self.image_size, self.image_size),
                method=Image.Resampling.LANCZOS,
                color=(127, 127, 127),
            )
            for image in images
        ]
        inputs = self.processor(
            images=padded,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)

        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                raise RuntimeError("vision backbone returned neither pooler_output nor last_hidden_state")
            pooled = hidden[:, 0]

        emb = pooled.detach().float().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.clip(norms, 1e-12, None)

    def embed_one(self, image: Image.Image) -> np.ndarray:
        return self.embed_images([image])[0]

    def embed_multicrop(self, image: Image.Image, tile_grid: int = 1, include_whole: bool = True) -> np.ndarray:
        """Optional ablation helper. Default tile_grid=1 returns the whole-image embedding only."""
        image = image.convert("RGB")
        crops: list[Image.Image] = [image] if include_whole else []
        if tile_grid > 1:
            width, height = image.size
            for row in range(tile_grid):
                for column in range(tile_grid):
                    left = round(column * width / tile_grid)
                    right = round((column + 1) * width / tile_grid)
                    top = round(row * height / tile_grid)
                    bottom = round((row + 1) * height / tile_grid)
                    crops.append(image.crop((left, top, right, bottom)))
        if not crops:
            crops = [image]
        vectors = self.embed_images(crops)
        fused = vectors.mean(axis=0)
        return (fused / max(float(np.linalg.norm(fused)), 1e-12)).astype(np.float32)


@dataclass
class BioCLIPEmbedder:
    """BioCLIP image encoder with full-specimen square padding before native normalization."""

    model_name: str = "hf-hub:imageomics/bioclip-2"
    device: str | None = None
    image_size: int = 224

    def __post_init__(self) -> None:
        self.device = self.device or best_device()
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError(
                "BioCLIP requires open_clip_torch; install requirements-embeddings.txt"
            ) from exc
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name)
        self.model.eval().to(self.device)

    def embed_images(self, images: Iterable[Image.Image]) -> np.ndarray:
        import torch

        images = [image.convert("RGB") for image in images]
        if not images:
            dimension = int(getattr(getattr(self.model, "visual", None), "output_dim", 768))
            return np.empty((0, dimension), dtype=np.float32)
        padded = [
            ImageOps.pad(
                image,
                (self.image_size, self.image_size),
                method=Image.Resampling.LANCZOS,
                color=(127, 127, 127),
            )
            for image in images
        ]
        batch = torch.stack([self.preprocess(image) for image in padded]).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_image(batch)
        vectors = features.detach().float().cpu().numpy().astype(np.float32)
        return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)

    def embed_one(self, image: Image.Image) -> np.ndarray:
        return self.embed_images([image])[0]

    def embed_multicrop(self, image: Image.Image, tile_grid: int = 1, include_whole: bool = True) -> np.ndarray:
        image = image.convert("RGB")
        crops: list[Image.Image] = [image] if include_whole else []
        if tile_grid > 1:
            width, height = image.size
            for row in range(tile_grid):
                for column in range(tile_grid):
                    crops.append(image.crop((
                        round(column * width / tile_grid),
                        round(row * height / tile_grid),
                        round((column + 1) * width / tile_grid),
                        round((row + 1) * height / tile_grid),
                    )))
        vectors = self.embed_images(crops or [image])
        fused = vectors.mean(axis=0)
        return (fused / max(float(np.linalg.norm(fused)), 1e-12)).astype(np.float32)


def create_embedder(backend: str, model_name: str, image_size: int):
    backend = backend.strip().lower()
    if backend == "dino":
        return DINOEmbedder(model_name, image_size=image_size)
    if backend == "bioclip":
        return BioCLIPEmbedder(model_name, image_size=image_size)
    raise ValueError(f"unsupported embedding backend: {backend}")


class FusionEmbedder:
    """Equal-weight late feature fusion for independently normalized encoders."""

    def __init__(self, encoder_configs: list[dict[str, Any]]):
        if len(encoder_configs) < 2:
            raise ValueError("fusion requires at least two encoder configurations")
        self.configs = encoder_configs
        self.encoders = [
            create_embedder(
                str(config["backend"]),
                str(config["backbone"]),
                int(config.get("image_size", 224)),
            )
            for config in encoder_configs
        ]

    @staticmethod
    def _join(vectors: list[np.ndarray]) -> np.ndarray:
        normalized = [
            vector.astype(np.float32) / max(float(np.linalg.norm(vector)), 1e-12)
            for vector in vectors
        ]
        fused = np.concatenate(normalized) / np.sqrt(len(normalized))
        return fused / max(float(np.linalg.norm(fused)), 1e-12)

    def embed_one(self, image: Image.Image) -> np.ndarray:
        return self._join([encoder.embed_one(image) for encoder in self.encoders])

    def embed_multicrop(self, image: Image.Image, tile_grid: int = 1, include_whole: bool = True) -> np.ndarray:
        return self._join([
            encoder.embed_multicrop(image, tile_grid=tile_grid, include_whole=include_whole)
            for encoder in self.encoders
        ])


def embedder_from_config(config: dict[str, Any]):
    if config.get("backend") == "fusion":
        return FusionEmbedder(list(config.get("encoders", [])))
    return create_embedder(
        str(config.get("backend", "dino")),
        str(config.get("backbone", config.get("model", "facebook/dinov3-vits16-pretrain-lvd1689m"))),
        int(config.get("image_size", 512)),
    )
