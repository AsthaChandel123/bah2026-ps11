"""FastAPI service for cross-modal satellite image retrieval.

Loads a prebuilt retrieval bundle (index + whitener + config; produced by
``python -m xsretrieval.cli build-index --out <dir>``) and serves:

* ``GET  /health``  — liveness + index stats.
* ``GET  /info``    — backbone / whitening / modality details.
* ``POST /query``   — multipart image upload (+ ``k`` and optional
  ``gallery_modality`` / ``query_modality`` form fields) → JSON top-k results.

Run
---
    # 1) build an index bundle once
    python -m xsretrieval.cli build-index --config configs/zero_shot.yaml --out artifacts/index

    # 2) serve it (bundle path via the XSRETRIEVAL_INDEX env var)
    XSRETRIEVAL_INDEX=artifacts/index uvicorn apps.api:app --host 0.0.0.0 --port 8000

    # 3) query it
    curl -F "file=@query.npy" -F "k=10" -F "gallery_modality=sar" \
        http://localhost:8000/query

All heavy imports are lazy / inside handlers so importing this module never
crashes when fastapi (or the compute stack) is absent — ``app`` is ``None`` in
that case and a clear message is printed when run directly.
"""

from __future__ import annotations

import io
import os
from typing import Any, Optional

# Default bundle directory; override with the XSRETRIEVAL_INDEX env var.
_DEFAULT_INDEX_DIR = os.environ.get("XSRETRIEVAL_INDEX", "artifacts/index")


def _build_app() -> Any:
    """Construct the FastAPI app, or return ``None`` if fastapi is unavailable."""
    try:
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import JSONResponse
    except Exception:  # pragma: no cover - optional dependency
        return None

    import numpy as np

    app = FastAPI(
        title="xsretrieval — Cross-Modal Satellite Image Retrieval",
        version="0.1.0",
        description="Retrieve semantically-matching satellite imagery across "
        "sensor modalities (optical / multispectral / SAR).",
    )

    # Lazy, cached engine state (loaded on first request).
    state: dict[str, Any] = {"engine": None, "config": None}

    def _get_engine():
        if state["engine"] is None:
            from xsretrieval.cli import _load_bundle

            index_dir = os.environ.get("XSRETRIEVAL_INDEX", _DEFAULT_INDEX_DIR)
            if not os.path.isdir(index_dir):
                raise HTTPException(
                    status_code=503,
                    detail=f"index bundle not found at {index_dir!r}; build it "
                    "with `xsretrieval build-index --out <dir>` and set "
                    "XSRETRIEVAL_INDEX.",
                )
            engine, config = _load_bundle(index_dir)
            state["engine"], state["config"] = engine, config
        return state["engine"], state["config"]

    def _decode_image(raw: bytes, filename: str):
        """Decode an uploaded file to a ``(C, H, W)`` float32 array."""
        if filename.endswith(".npy"):
            arr = np.load(io.BytesIO(raw)).astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(arr.shape[0], 1, 1)
            return arr
        from PIL import Image

        img = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1))  # HWC -> CHW

    @app.get("/health")
    def health() -> dict:
        """Liveness probe + gallery size if the bundle is loaded."""
        engine = state["engine"]
        n = 0
        backend = "unloaded"
        if engine is not None and engine.get_index() is not None:
            n = len(engine.get_index())
            backend = "faiss" if engine.get_index().uses_faiss else "numpy"
        return {"status": "ok", "gallery_size": n, "index_backend": backend}

    @app.get("/info")
    def info() -> dict:
        """Backbone / whitening / modality metadata for the loaded bundle."""
        engine, config = _get_engine()
        idx = engine.get_index()
        return {
            "backbone": config.backbone.name,
            "backbone_class": type(engine.backbone).__name__,
            "embed_dim": engine.embed_dim,
            "whitening": engine.whitener is not None,
            "gallery_size": len(idx) if idx is not None else 0,
            "index_backend": "faiss" if (idx and idx.uses_faiss) else "numpy",
            "modalities": config.data.modalities,
        }

    @app.post("/query")
    async def query(
        file: "UploadFile" = File(...),
        k: int = Form(10),
        query_modality: Optional[str] = Form(None),
        gallery_modality: Optional[str] = Form(None),
    ) -> "JSONResponse":
        """Retrieve the top-``k`` gallery items for an uploaded query image."""
        from xsretrieval.data.modalities import Modality, Sample

        engine, config = _get_engine()
        raw = await file.read()
        try:
            arr = _decode_image(raw, file.filename or "upload")
        except Exception as exc:  # pragma: no cover - input dependent
            raise HTTPException(status_code=400, detail=f"could not decode image: {exc}")

        q_mod = Modality(query_modality) if query_modality else Modality(config.data.modalities[0])
        g_mod = Modality(gallery_modality) if gallery_modality else None
        sample = Sample(id="__query__", image=arr, modality=q_mod, label=-1)
        emb = engine.encode([sample])
        hits = engine.query(
            emb[0] if emb.ndim == 2 else emb,
            k=int(k),
            gallery_modality=g_mod,
            exclude_same_location=False,
        )
        results = [
            {
                "rank": h["rank"],
                "id": str(h["id"]),
                "modality": str(h["modality"]),
                "label": int(h["label"]),
                "score": float(h["score"]),
            }
            for h in hits
        ]
        return JSONResponse(
            {
                "query_modality": q_mod.value,
                "gallery_modality": g_mod.value if g_mod else "all",
                "k": int(k),
                "results": results,
            }
        )

    return app


# Module-level app (None if fastapi is not installed). uvicorn imports this.
app = _build_app()


if __name__ == "__main__":  # pragma: no cover
    if app is None:
        print(
            "FastAPI is not installed. Install the serving extra:\n"
            "  pip install fastapi uvicorn python-multipart pillow\n"
            "Then run:\n"
            "  XSRETRIEVAL_INDEX=artifacts/index uvicorn apps.api:app --port 8000"
        )
        raise SystemExit(1)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
