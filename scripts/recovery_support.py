"""Small, local-only recovery primitives; never delete a user's completed data."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


def nonempty(path: str | Path) -> bool:
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def backup_file(path: Path) -> Path | None:
    if not nonempty(path):
        return None
    sha = digest(path)
    target = path.parent / "recovery_backups" / f"{path.name}.{sha[:16]}.bak"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        pending = target.with_name(target.name + ".pending")
        shutil.copy2(path, pending)
        if digest(pending) != sha:
            raise RuntimeError(f"Backup verification failed: {path}; original untouched")
        pending.replace(target)
    elif digest(target) != sha:
        raise RuntimeError(f"Existing backup is damaged: {target}; original untouched")
    print(f"Backup verified: {target}", flush=True)
    return target


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)


def checkpoint_bioscan(layout: dict[str, Path]) -> None:
    root = layout["bioscan"]
    names = ["diptera_30k_selection.csv", "diptera_30k_downloaded.csv",
             "diptera_added_families_topup_selection.csv",
             "diptera_added_families_topup_selection_report.json",
             "diptera_added_families_topup_downloaded.csv",
             "diptera_added_families_topup_download_report.json",
             "diptera_added_families_topup_progress.json"]
    saved = []
    for path in [layout["bioscan_manifest"], *(root / name for name in names)]:
        backup = backup_file(path)
        if backup:
            saved.append({"original": str(path), "backup": str(backup), "sha256": digest(path)})
    atomic_json(layout["poc"] / "bioscan_recovery_checkpoint.json", {"files": saved})
    print(f"Checkpoint: {len(saved)} files backed up. Image folders were not changed.", flush=True)


def reusable_topup(selection: Path, report_path: Path, families: list[str], minimum: int) -> bool:
    """Accept the legacy successful selection too; never silently overwrite it."""
    if not selection.exists() and not report_path.exists():
        return False
    import pandas as pd
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(selection, dtype=str, keep_default_na=False)
        required = {"processid", "family", "archive_group", "archive_member"}
        if not required.issubset(frame.columns) or frame.empty:
            raise ValueError("missing columns or empty selection")
        if frame.processid.eq("").any() or frame.processid.duplicated().any():
            raise ValueError("empty or duplicated specimen IDs")
        if not frame.archive_group.isin(["train", "eval", "pretrain"]).all() or frame.archive_member.eq("").any():
            raise ValueError("missing/invalid archive locations")
        wanted = {f.casefold() for f in families}
        counts = frame.family.str.casefold().value_counts()
        if set(counts.index) != wanted or any(counts.get(f, 0) < minimum for f in wanted):
            raise ValueError("family quotas do not match")
        if report.get("min_per_family") != minimum or {f.casefold() for f in report.get("families", [])} != wanted:
            raise ValueError("saved settings do not match")
        if report.get("topup_selection_rows") != len(frame):
            raise ValueError("selection/report row counts differ")
        if any(value.get("shortfall", 0) for value in report.get("by_family", {}).values()):
            raise ValueError("saved selection has a shortfall")
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"Saved BIOSCAN selection needs inspection ({exc}). It was NOT overwritten: {selection}") from exc
    print(f"REUSING SAVED TOP-UP: {len(frame)} rows; no metadata rescan: {selection}", flush=True)
    return True


def publish_bioscan(merged: Path, pending: Path, downloaded: Path, normalized: Path) -> None:
    import pandas as pd
    updated = pd.read_parquet(pending)
    if updated.empty or "source_record_id" not in updated:
        raise RuntimeError("New BIOSCAN manifest is empty/invalid; original untouched")
    if nonempty(normalized):
        old = pd.read_parquet(normalized)
        updated = pd.concat([old, updated], ignore_index=True).drop_duplicates("source_record_id", keep="last")
        if not set(old.source_record_id).issubset(set(updated.source_record_id)):
            raise RuntimeError("Preservation check failed; original untouched")
        updated.to_parquet(pending, index=False)
    backup_file(downloaded)
    backup_file(normalized)
    merged.replace(downloaded)
    pending.replace(normalized)
    print(f"BIOSCAN saved: {len(updated)} specimen rows. Previous IDs preserved.", flush=True)
