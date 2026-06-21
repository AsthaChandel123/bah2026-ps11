"""Interactive demo for cross-modal satellite image retrieval.

Upload a query image, choose the gallery modality, and see the top-5 / top-10
retrieved items with their similarity scores. Uses Gradio when available; falls
back to a pure-CLI prompt loop when Gradio is not installed, so the demo is
always runnable.

Run
---
    # build an index bundle first
    python -m xsretrieval.cli build-index --config configs/zero_shot.yaml --out artifacts/index

    # Gradio UI (if gradio installed):
    XSRETRIEVAL_INDEX=artifacts/index python apps/demo.py

    # CLI fallback (no gradio): same command prints a prompt loop.

Importing this module never crashes when gradio / the compute stack is absent.
"""

from __future__ import annotations

import os
from typing import Any, Optional

_DEFAULT_INDEX_DIR = os.environ.get("XSRETRIEVAL_INDEX", "artifacts/index")


def load_engine(index_dir: Optional[str] = None):
    """Load the retrieval engine + config from a persisted bundle directory."""
    from xsretrieval.cli import _load_bundle

    index_dir = index_dir or os.environ.get("XSRETRIEVAL_INDEX", _DEFAULT_INDEX_DIR)
    if not os.path.isdir(index_dir):
        raise FileNotFoundError(
            f"index bundle not found at {index_dir!r}. Build one first:\n"
            "  python -m xsretrieval.cli build-index --config configs/zero_shot.yaml "
            "--out artifacts/index"
        )
    return _load_bundle(index_dir)


def _to_chw(image, config, modality):
    """Coerce a demo input (file path / numpy HWC / .npy) to a (C, H, W) array."""
    import numpy as np

    if isinstance(image, str):
        if image.endswith(".npy"):
            arr = np.load(image).astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(arr.shape[0], 1, 1)
            return arr
        from xsretrieval.data.datasets import load_image_any

        arr = load_image_any(image).astype(np.float32)
    else:
        arr = np.asarray(image, dtype=np.float32)
        if arr.max() > 1.5:  # 0-255 image
            arr = arr / 255.0
    if arr.ndim == 3 and arr.shape[2] <= 16 and arr.shape[0] > arr.shape[2]:
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    elif arr.ndim == 2:
        arr = arr[None, :, :]
    return arr


def retrieve(engine, config, image, query_modality: str, gallery_modality: str, k: int):
    """Run a single retrieval and return a list of result dicts."""
    from xsretrieval.data.modalities import Modality, Sample

    q_mod = Modality(query_modality) if query_modality else Modality(config.data.modalities[0])
    g_mod = None if (not gallery_modality or gallery_modality == "all") else Modality(gallery_modality)
    arr = _to_chw(image, config, q_mod)
    sample = Sample(id="__query__", image=arr, modality=q_mod, label=-1)
    emb = engine.encode([sample])
    hits = engine.query(
        emb[0] if emb.ndim == 2 else emb,
        k=int(k),
        gallery_modality=g_mod,
        exclude_same_location=False,
    )
    return hits


def _launch_gradio(engine, config) -> bool:
    """Launch the Gradio UI. Returns ``False`` if gradio is unavailable."""
    try:
        import gradio as gr
    except Exception:  # pragma: no cover - optional dependency
        return False

    modalities = list(config.data.modalities)
    gallery_choices = ["all", *modalities]

    def _fn(image, query_modality, gallery_modality, k):
        if image is None:
            return "Upload a query image (or a .npy embedding) to retrieve."
        hits = retrieve(engine, config, image, query_modality, gallery_modality, int(k))
        lines = [
            f"{'rank':<5}{'id':<26}{'modality':<14}{'label':<7}{'score':>8}",
            "-" * 60,
        ]
        for h in hits:
            lines.append(
                f"{h['rank']:<5}{str(h['id']):<26}{str(h['modality']):<14}"
                f"{h['label']:<7}{h['score']:>8.4f}"
            )
        return "\n".join(lines)

    with gr.Blocks(title="xsretrieval — Cross-Modal Satellite Retrieval") as demo:
        gr.Markdown(
            "# Cross-Modal Satellite Image Retrieval\n"
            "Upload a query image, pick the **gallery modality**, and retrieve "
            "semantically-matching imagery across sensors (optical / MS / SAR)."
        )
        with gr.Row():
            inp = gr.Image(label="Query image", type="numpy")
            with gr.Column():
                q_mod = gr.Dropdown(modalities, value=modalities[0], label="Query modality")
                g_mod = gr.Dropdown(gallery_choices, value="all", label="Gallery modality")
                k = gr.Slider(1, 20, value=10, step=1, label="Top-k")
                btn = gr.Button("Retrieve", variant="primary")
        out = gr.Textbox(label="Results", lines=14)
        btn.click(_fn, inputs=[inp, q_mod, g_mod, k], outputs=out)

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
    return True


def _cli_loop(engine, config) -> None:
    """Pure-CLI fallback: prompt for image paths and print results."""
    print("xsretrieval demo (CLI fallback — gradio not installed)")
    print(f"Loaded bundle: backbone={config.backbone.name} "
          f"modalities={config.data.modalities}")
    print("Enter a query image path (or .npy embedding); blank line to quit.\n")
    while True:
        try:
            path = input("query image> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not path:
            break
        if not os.path.exists(path):
            print(f"  not found: {path}")
            continue
        gm = input("  gallery modality [all]> ").strip() or "all"
        qm = config.data.modalities[0]
        try:
            hits = retrieve(engine, config, path, qm, gm, 10)
        except Exception as exc:  # pragma: no cover - input dependent
            print(f"  error: {exc}")
            continue
        print(f"  {'rank':<5}{'id':<26}{'modality':<14}{'label':<7}{'score':>8}")
        for h in hits:
            print(f"  {h['rank']:<5}{str(h['id']):<26}{str(h['modality']):<14}"
                  f"{h['label']:<7}{h['score']:>8.4f}")
        print()


def main() -> int:
    """Launch the demo (Gradio if available, else CLI loop)."""
    engine, config = load_engine()
    if not _launch_gradio(engine, config):
        _cli_loop(engine, config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
