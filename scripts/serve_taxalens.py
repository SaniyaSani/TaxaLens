from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the TaxaLens evidence-first web workbench")
    parser.add_argument("--model-dir", required=True, help="Directory containing classifiers.joblib and retrieval artifacts")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    required = ["classifiers.joblib", "retrieval_vectors.npy", "retrieval_metadata.csv"]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        raise SystemExit(f"STOP: model directory is incomplete: {model_dir}; missing {', '.join(missing)}")

    os.environ["DIPTERA_MODEL_DIR"] = str(model_dir)
    import uvicorn

    uvicorn.run("app.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
