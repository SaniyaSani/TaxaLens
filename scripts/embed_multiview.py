#!/usr/bin/env python3
"""Create whole-fly + high-resolution tile embeddings and fuse all views of each specimen."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diptera_id.corpus.io import load_manifest


def crops_for(image: Image.Image, tile_grid: int, include_whole: bool) -> list[Image.Image]:
    image = image.convert("RGB")
    crops = [image] if include_whole else []
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
    return crops or [image]


def normalized_mean(vectors: np.ndarray) -> np.ndarray:
    fused = vectors.mean(axis=0)
    return (fused / max(float(np.linalg.norm(fused)), 1e-12)).astype(np.float32)


def fuse_specimens(frame: pd.DataFrame, vectors: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    if len(frame) != len(vectors):
        raise ValueError("view manifest and embeddings have different row counts")
    frame = frame.copy()
    if "record_id" not in frame:
        frame["record_id"] = [f"view-{index}" for index in range(len(frame))]
    record_keys = frame["record_id"].astype(str).str.strip()
    if record_keys.eq("").any():
        record_keys = record_keys.where(
            record_keys.ne(""),
            pd.Series([f"row-{index}" for index in range(len(frame))], index=frame.index),
        )
    if "specimen_group_id" in frame:
        specimen_keys = frame["specimen_group_id"].astype(str).str.strip()
        # Blank specimen ids are separate records, never one giant blank specimen.
        group_keys = specimen_keys.where(specimen_keys.ne(""), "record:" + record_keys)
    else:
        group_keys = "record:" + record_keys
    frame["__embedding_specimen_key"] = group_keys

    rows: list[dict] = []
    fused_vectors: list[np.ndarray] = []
    for group, indices in frame.groupby("__embedding_specimen_key", sort=True).indices.items():
        selected = np.asarray(indices, dtype=int)
        row = frame.iloc[selected[0]].to_dict()
        views = sorted({str(value) for value in frame.iloc[selected].get("view_type", pd.Series(["habitus"])).tolist() if str(value)})
        row["view_count"] = len(selected)
        row["available_views"] = ",".join(views) or "habitus"
        row["embedding_view_ids"] = ",".join(sorted(
            str(value) for value in frame.iloc[selected]["record_id"].tolist()
        ))
        existing_specimen = str(row.get("specimen_group_id", "")).strip()
        row["specimen_group_id"] = existing_specimen or str(group)
        split_groups = {
            str(value).strip()
            for value in frame.iloc[selected].get("split_group", pd.Series(dtype=str)).tolist()
            if str(value).strip()
        }
        if len(split_groups) > 1:
            raise ValueError(f"conflicting split_group values within specimen {group}")
        row["split_group"] = next(iter(split_groups), str(group))
        if "split" in frame:
            splits = {str(value).strip() for value in frame.iloc[selected]["split"] if str(value).strip()}
            if len(splits) > 1:
                raise ValueError(f"conflicting dataset splits within specimen {group}")
        for rank in ("family", "genus", "species"):
            if rank not in frame.columns:
                continue
            values = {str(value) for value in frame.iloc[selected][rank].tolist() if str(value)}
            if len(values) > 1:
                row[rank] = ""
                row["eligible_supervised"] = False
                row["exclusion_reason"] = f"conflicting_{rank}_within_specimen"
        rows.append(row)
        fused_vectors.append(normalized_mean(vectors[selected]))
    result = pd.DataFrame(rows).drop(columns=["__embedding_specimen_key"], errors="ignore")
    return result, np.stack(fused_vectors).astype(np.float32)


def main() -> None:
    from diptera_id.embedding import DINOEmbedder

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", default="models_microdiptera")
    parser.add_argument("--model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--tile-grid", type=int, default=1, help="1 = whole image only; set 2+ only for an explicit ablation")
    parser.add_argument("--batch-size", type=int, default=4, help="Specimens per GPU batch; tiles are produced only when --tile-grid > 1")
    parser.add_argument("--no-whole", action="store_true")
    args = parser.parse_args()
    from diptera_id.embedding import validate_image_size
    try:
        validate_image_size(args.model, args.image_size)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.tile_grid < 1 or args.tile_grid > 4:
        raise SystemExit("tile-grid must be between 1 and 4")

    frame = load_manifest(args.manifest).fillna("")
    frame = frame[frame["local_path"].astype(str).map(lambda value: Path(value).is_file())].reset_index(drop=True)
    if frame.empty:
        raise SystemExit("No local images found in manifest")
    if "view_type" not in frame.columns:
        frame["view_type"] = "habitus"
    frame.loc[frame["view_type"].astype(str).eq(""), "view_type"] = "habitus"

    embedder = DINOEmbedder(args.model, image_size=args.image_size)
    kept_rows: list[int] = []
    view_vectors: list[np.ndarray] = []
    include_whole = not args.no_whole
    for start in range(0, len(frame), args.batch_size):
        chunk = frame.iloc[start:start + args.batch_size]
        flat_crops: list[Image.Image] = []
        counts: list[int] = []
        valid_indices: list[int] = []
        for index, row in chunk.iterrows():
            try:
                image = Image.open(row["local_path"]).convert("RGB")
                crops = crops_for(image, args.tile_grid, include_whole)
            except Exception as exc:
                print(f"skip {row['local_path']}: {exc}")
                continue
            flat_crops.extend(crops)
            counts.append(len(crops))
            valid_indices.append(index)
        if flat_crops:
            embedded = embedder.embed_images(flat_crops)
            cursor = 0
            for index, count in zip(valid_indices, counts):
                view_vectors.append(normalized_mean(embedded[cursor:cursor + count]))
                kept_rows.append(index)
                cursor += count
        print(f"embedded views: {min(start + args.batch_size, len(frame))}/{len(frame)}")

    if not view_vectors:
        raise SystemExit("No images could be embedded")
    view_frame = frame.loc[kept_rows].reset_index(drop=True)
    view_array = np.stack(view_vectors).astype(np.float32)
    specimen_frame, specimen_array = fuse_specimens(view_frame, view_array)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "view_embeddings.npy", view_array)
    view_frame.to_csv(out / "embedded_views_manifest.csv", index=False)
    np.save(out / "embeddings.npy", specimen_array)
    specimen_frame.to_csv(out / "embedded_manifest.csv", index=False)
    config = {
        "backbone": args.model,
        "image_size": args.image_size,
        "tile_grid": args.tile_grid,
        "include_whole": include_whole,
        "view_images": len(view_frame),
        "fused_specimens": len(specimen_frame),
    }
    (out / "embedding_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"saved {len(view_frame)} views fused into {len(specimen_frame)} specimens -> {out}")


if __name__ == "__main__":
    main()
