from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path
from typing import Iterator

import pandas as pd

from .schema import MASTER_COLUMNS, finalize_record


def infer_separator(path: str | Path) -> str:
    name = str(path).lower()
    return "\t" if name.endswith((".tsv", ".txt", ".tsv.gz", ".txt.gz")) else ","


def iter_table(path: str | Path, chunksize: int = 100_000) -> Iterator[pd.DataFrame]:
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz")):
        yield from pd.read_csv(path, sep=infer_separator(path), dtype=str, keep_default_na=False, chunksize=chunksize, low_memory=False)
        return
    if name.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit("Parquet input requires pyarrow: pip install pyarrow") from exc
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=chunksize):
            yield batch.to_pandas()
        return
    if name.endswith((".jsonl", ".ndjson")):
        buffer: list[dict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    buffer.append(json.loads(line))
                if len(buffer) >= chunksize:
                    yield pd.DataFrame(buffer)
                    buffer = []
        if buffer:
            yield pd.DataFrame(buffer)
        return
    if name.endswith(".json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("results", "items", "data", "digitalSpecimen"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
            else:
                payload = [payload]
        for start in range(0, len(payload), chunksize):
            yield pd.DataFrame(payload[start:start + chunksize])
        return
    raise SystemExit(f"Unsupported input format: {path}")


class ManifestWriter:
    """Append normalized chunks to CSV(.gz) or Parquet without keeping all rows in RAM."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._pending = self.path.with_name(self.path.name + f".{uuid.uuid4().hex}.pending")
        self.rows_written = 0
        self._parquet_writer = None
        self._csv_header = True

    def write(self, rows: list[dict] | pd.DataFrame) -> None:
        if not isinstance(rows, pd.DataFrame):
            rows = pd.DataFrame([finalize_record(row) for row in rows])
        if rows.empty:
            return
        for column in MASTER_COLUMNS:
            if column not in rows.columns:
                rows[column] = ""
        rows = rows[MASTER_COLUMNS].copy()
        self.rows_written += len(rows)

        if self.path.suffix.lower() == ".parquet":
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise SystemExit("Parquet output requires pyarrow: pip install pyarrow") from exc
            # Stable string schema avoids chunk-to-chunk type drift in heterogeneous sources.
            normalized = rows.astype(str)
            table = pa.Table.from_pandas(normalized, preserve_index=False)
            if self._parquet_writer is None:
                self._parquet_writer = pq.ParquetWriter(self._pending, table.schema, compression="zstd")
            self._parquet_writer.write_table(table)
        else:
            rows.to_csv(
                self._pending,
                index=False,
                mode="w" if self._csv_header else "a",
                header=self._csv_header,
                compression="gzip" if str(self.path).endswith(".gz") else None,
                quoting=csv.QUOTE_MINIMAL,
            )
            self._csv_header = False

    def close(self, commit: bool = True) -> None:
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None
        if commit and self.rows_written and self._pending.exists():
            self._pending.replace(self.path)

    def __enter__(self) -> "ManifestWriter":
        return self

    def __exit__(self, exc_type, *_exc) -> None:
        self.close(commit=exc_type is None)


def load_manifest(path: str | Path) -> pd.DataFrame:
    chunks = list(iter_table(path))
    return pd.concat(chunks, ignore_index=True, sort=False) if chunks else pd.DataFrame(columns=MASTER_COLUMNS)
