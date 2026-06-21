#!/usr/bin/env python
"""One-command, hands-off **real-data proof** for ``xsretrieval`` (BAH 2026 PS-11).

This script is the automated entry point that turns the framework into a genuine
result on **real satellite imagery** with a **real foundation backbone** — no
synthetic substrate, no placeholder numbers. It orchestrates the full pipeline
through the package's real Python API (it does **not** depend on CLI flags that
do not exist):

    download data  →  load real Samples  →  class-balanced query/gallery split
    →  build a RetrievalEngine (real backbone + per-modality whitening)
    →  [optional] train the projection head  →  encode + index (faiss/numpy)
    →  evaluate (F1@5/@10 same- & cross-modal + latency)  →  save artifacts
    →  write a Markdown report

Design goals
------------
* **Robust / hands-off.** Missing data is downloaded automatically (EuroSAT). A
  backbone whose weights cannot be fetched gracefully falls back to the numpy
  ``FallbackBackbone`` and the report records *which* backbone actually ran.
* **CPU-sane.** ``--subset`` caps the image count and encoding is batched, so a
  real run finishes in minutes on a CPU box; ``--device cuda`` is honored when
  available.
* **Deterministic** given ``--seed``.
* **Honest.** Cross-modal numbers are only reported when a backbone that *treats
  multispectral differently from RGB* actually loads (otherwise RGB≈MS is a
  trivially-inflated non-test and cross-modal real proof is explicitly deferred
  to a multispectral-capable backbone — DOFA/CROMA — on GPU/SEN12MS).

Example (the guaranteed real same-modal proof, CPU)::

    python scripts/run_real_proof.py --dataset eurosat --backbone auto \
        --device cpu --subset 2000 --gallery-per-class 10 --query-per-class 5 \
        --no-train

On a GPU box, the full cross-modal proof toward F1≥0.8::

    python scripts/run_real_proof.py --dataset sen12ms --backbone dofa \
        --device cuda --train --epochs 30 --gallery-per-class 8

The public orchestration entry point is :func:`run_real_proof`, which returns a
plain results ``dict`` (used by the light, network-free wiring test).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Any, Optional

# Make the package importable when run as ``python scripts/run_real_proof.py``
# from a fresh checkout (mirrors tests/conftest.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402

logger = logging.getLogger("run_real_proof")

# Real backbones to try, in order, for --backbone auto. Each entry is
# (registry_name, kwargs). DINOv2-small and OpenCLIP ViT-B/32 are small enough
# to download + run on a CPU box. They are RGB-only (pseudo-RGB adaptation for
# MS/SAR), so they do NOT constitute a real cross-modal MS test (see
# _backbone_is_multispectral_aware). DOFA/CROMA are tried last as the genuine
# multispectral-capable options (heavier; mainly for GPU/SEN12MS runs).
_AUTO_BACKBONES: list[tuple[str, dict[str, Any]]] = [
    ("dinov2", {"model_name": "facebook/dinov2-small"}),
    ("openclip", {"arch": "ViT-B-32", "pretrained_tag": "laion2b_s34b_b79k"}),
    ("dofa", {}),
    ("dinov2", {"model_name": "facebook/dinov2-with-registers-base"}),
]

# Backbone classes that genuinely consume >3 spectral bands differently from RGB
# (i.e. a real optical↔multispectral test). RGB-only encoders (DINOv2, CLIP,
# timm) reduce MS to pseudo-RGB and are therefore NOT in this set.
_MULTISPECTRAL_AWARE_CLASSES = {"DOFABackbone", "CROMABackbone"}


# ---------------------------------------------------------------------------
# Small logging helper with timing
# ---------------------------------------------------------------------------
class _Step:
    """Context manager that logs a pipeline step with wall-clock timing."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.t0 = 0.0
        self.secs = 0.0

    def __enter__(self) -> "_Step":
        self.t0 = time.time()
        logger.info("▶ %s ...", self.name)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.secs = time.time() - self.t0
        logger.info("✓ %s (%.1fs)", self.name, self.secs)


# ---------------------------------------------------------------------------
# Device / environment
# ---------------------------------------------------------------------------
def resolve_device(requested: str) -> str:
    """Resolve ``auto|cuda|cpu`` to a concrete torch device string.

    ``auto`` → cuda when ``torch.cuda.is_available()`` else cpu. A requested
    ``cuda`` that is unavailable degrades to cpu with a warning. If torch is not
    installed at all, returns ``"cpu"`` (the numpy path still works).
    """
    req = (requested or "auto").lower()
    try:
        import torch  # local, lazy

        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    if req == "auto":
        return "cuda" if cuda_ok else "cpu"
    if req.startswith("cuda") and not cuda_ok:
        logger.warning("device 'cuda' requested but not available; using cpu.")
        return "cpu"
    return req


def env_report() -> dict[str, Any]:
    """Collect a small environment report (torch/faiss/cuda availability)."""
    rep: dict[str, Any] = {"python": sys.version.split()[0], "platform": sys.platform}
    try:
        import torch

        rep["torch"] = torch.__version__
        rep["cuda_available"] = bool(torch.cuda.is_available())
        rep["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:
        rep["torch"] = None
        rep["cuda_available"] = False
        rep["cuda_device"] = None
    try:
        import faiss  # noqa: F401

        rep["faiss"] = True
    except Exception:
        rep["faiss"] = False
    for mod in ("transformers", "open_clip", "PIL", "rasterio"):
        try:
            __import__(mod)
            rep[mod] = True
        except Exception:
            rep[mod] = False
    return rep


# ---------------------------------------------------------------------------
# Data: download + load
# ---------------------------------------------------------------------------
def _eurosat_dir(data_root: str, bands: str) -> Optional[str]:
    """Return the EuroSAT class-folder root under *data_root*, or ``None``.

    The official archives extract to a few different layouts depending on the
    mirror (``EuroSAT/2750/<class>``, ``2750/<class>``, or ``<class>`` directly).
    We search for the directory that actually contains per-class sub-folders.
    """
    base = os.path.join(data_root, "eurosat_rgb" if bands == "rgb" else "eurosat_ms")
    if not os.path.isdir(base):
        return None
    candidates = [
        base,
        os.path.join(base, "2750"),
        os.path.join(base, "EuroSAT"),
        os.path.join(base, "EuroSAT", "2750"),
        os.path.join(base, "EuroSATallBands"),
        os.path.join(base, "ds", "images", "remote_sensing", "otherDatasets",
                     "sentinel_2", "tif"),
    ]
    for cand in candidates:
        if _looks_like_classfolder(cand):
            return cand
    # Fallback: walk to find the first directory whose children are class dirs
    # holding image files.
    for dirpath, dirnames, _filenames in os.walk(base):
        if _looks_like_classfolder(dirpath):
            return dirpath
    return None


def _looks_like_classfolder(path: str) -> bool:
    """True if *path* has ≥2 sub-dirs that each contain raster files."""
    if not os.path.isdir(path):
        return False
    subdirs = [
        d for d in sorted(os.listdir(path))
        if os.path.isdir(os.path.join(path, d))
    ]
    good = 0
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    for d in subdirs:
        dp = os.path.join(path, d)
        try:
            files = os.listdir(dp)
        except OSError:
            continue
        if any(f.lower().endswith(exts) for f in files):
            good += 1
        if good >= 2:
            return True
    return False


def ensure_eurosat(data_root: str, bands: str) -> Optional[str]:
    """Ensure EuroSAT (*bands* = ``rgb``|``all``) is on disk; download if absent.

    Returns the resolved class-folder root, or ``None`` if the data could not be
    obtained (the caller then falls back to synthetic-free… actually to the
    pipeline's own synthetic fallback via load_samples, which we avoid by
    erroring out so the proof never silently goes synthetic).
    """
    key = "rgb" if bands == "rgb" else "all"
    found = _eurosat_dir(data_root, key)
    if found is not None:
        logger.info("EuroSAT (%s) already present at %s", key, found)
        return found

    os.makedirs(data_root, exist_ok=True)
    target = os.path.join(data_root, "eurosat_rgb" if key == "rgb" else "eurosat_ms")
    logger.info("EuroSAT (%s) not found; downloading into %s", key, target)
    try:
        from scripts.download_data import fetch_eurosat
    except Exception:
        # When run as a top-level script the package import name differs.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from download_data import fetch_eurosat  # type: ignore

    rc = fetch_eurosat(target, bands=key)
    if rc != 0:
        logger.error("EuroSAT download failed (rc=%s).", rc)
        return None
    return _eurosat_dir(data_root, key)


def load_eurosat_samples(
    data_root: str,
    want_multispectral: bool,
    subset: int,
    seed: int,
) -> tuple[list, dict[str, Any]]:
    """Load real EuroSAT samples (RGB always; +MS with derived RGB if requested).

    Returns ``(samples, info)``. When *want_multispectral* is true the all-bands
    (13-band) EuroSAT is used with ``derive_rgb=True`` so each patch yields a
    co-registered (multispectral, optical_rgb) pair sharing a ``location_id`` —
    the cross-modal positive structure. Otherwise the small RGB set is used.

    The subset is applied **class-balanced** (round-robin across classes) so the
    downstream class-balanced gallery split is well-populated for every class.
    """
    from xsretrieval.data.datasets import EuroSATDataset

    info: dict[str, Any] = {}
    if want_multispectral:
        root = ensure_eurosat(data_root, "all")
        if root is None:
            raise FileNotFoundError(
                "EuroSAT all-bands could not be downloaded; cannot run the "
                "multispectral cross-modal proof."
            )
        ds = EuroSATDataset(root, variant="ms", derive_rgb=True)
        info["variant"] = "ms+derived_rgb"
    else:
        root = ensure_eurosat(data_root, "rgb")
        if root is None:
            raise FileNotFoundError(
                "EuroSAT RGB could not be downloaded; cannot run the real proof "
                "(refusing to silently fall back to synthetic data)."
            )
        ds = EuroSATDataset(root, variant="rgb")
        info["variant"] = "rgb"

    info["root"] = root
    info["classes"] = list(getattr(ds, "classes", []))

    # Materialise per (class, modality) so we can subset class-balanced.
    all_samples = ds.to_list()
    samples = _subset_class_balanced(all_samples, subset, seed)
    info["n_total_on_disk"] = len(all_samples)
    info["n_loaded"] = len(samples)
    return samples, info


def _subset_class_balanced(samples: list, subset: int, seed: int) -> list:
    """Round-robin subset *samples* to ≤ *subset*, balanced over (label, modality).

    ``subset <= 0`` means keep everything. Items are grouped by (label, modality)
    and drawn round-robin so every class — in every modality present — is
    represented before any class is over-sampled. Deterministic given *seed*.
    """
    if subset is None or subset <= 0 or len(samples) <= subset:
        return list(samples)
    rng = np.random.default_rng(seed)
    groups: dict[tuple, list] = defaultdict(list)
    for s in samples:
        mod = s.modality.value if hasattr(s.modality, "value") else str(s.modality)
        groups[(int(s.label), mod)].append(s)
    # Shuffle within each group for an unbiased subset.
    keys = sorted(groups.keys())
    for k in keys:
        idx = np.arange(len(groups[k]))
        rng.shuffle(idx)
        groups[k] = [groups[k][i] for i in idx]
    out: list = []
    cursors = {k: 0 for k in keys}
    while len(out) < subset:
        progressed = False
        for k in keys:
            if cursors[k] < len(groups[k]):
                out.append(groups[k][cursors[k]])
                cursors[k] += 1
                progressed = True
                if len(out) >= subset:
                    break
        if not progressed:
            break
    return out


# ---------------------------------------------------------------------------
# Query/gallery split (class-balanced, R_q ≈ K)
# ---------------------------------------------------------------------------
def class_balanced_split(
    samples: list,
    gallery_per_class: int,
    query_per_class: int,
    seed: int,
) -> tuple[list, list]:
    """Build a per-modality, class-balanced (query, gallery) split.

    Each modality is split independently (so every modality appears in both
    query and gallery — required to fill the cross-modal matrix). Within a
    modality, each class contributes **exactly** ``gallery_per_class`` gallery
    items (so ``R_q ≈ gallery_per_class``, the F1@K lever) and up to
    ``query_per_class`` query items. Classes with too few samples contribute
    what they can. Deterministic given *seed*.
    """
    from xsretrieval.data.modalities import Modality

    by_mod: dict[Any, list] = defaultdict(list)
    for s in samples:
        m = s.modality if isinstance(s.modality, Modality) else Modality(s.modality)
        by_mod[m].append(s)

    queries: list = []
    gallery: list = []
    for mod, items in by_mod.items():
        q, g = _split_one_modality(
            items, gallery_per_class, query_per_class, seed
        )
        queries.extend(q)
        gallery.extend(g)
    return queries, gallery


def _split_one_modality(
    items: list,
    gallery_per_class: int,
    query_per_class: int,
    seed: int,
) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    by_class: dict[int, list] = defaultdict(list)
    for s in items:
        by_class[int(s.label)].append(s)
    queries: list = []
    gallery: list = []
    for label in sorted(by_class.keys()):
        group = by_class[label]
        idx = np.arange(len(group))
        rng.shuffle(idx)
        g_n = min(gallery_per_class, max(0, len(group) - 1))
        q_n = min(query_per_class, len(group) - g_n)
        g_idx = idx[:g_n]
        q_idx = idx[g_n : g_n + q_n]
        gallery.extend(group[i] for i in g_idx)
        queries.extend(group[i] for i in q_idx)
    return queries, gallery


# ---------------------------------------------------------------------------
# Backbone resolution (with graceful fallback)
# ---------------------------------------------------------------------------
def build_backbone(name: str, embed_dim: int, device: str) -> tuple[Any, dict[str, Any]]:
    """Build the real backbone, falling back to numpy ``FallbackBackbone``.

    For ``name == "auto"`` the candidates in ``_AUTO_BACKBONES`` are tried in
    order; the first that loads a *real* (non-fallback) class wins. For an
    explicit name, ``get_backbone`` is called directly (which itself falls back
    to numpy on failure). Returns ``(backbone, info)`` where ``info`` records the
    requested name, the actual class, whether it fell back, and whether it is
    multispectral-aware.
    """
    from xsretrieval.models import get_backbone

    info: dict[str, Any] = {"requested": name, "device": device}

    if name.lower() != "auto":
        bb = get_backbone(name, embed_dim=embed_dim, device=device)
        return bb, _finalize_backbone_info(info, bb)

    last: Any = None
    for cand_name, cand_kwargs in _AUTO_BACKBONES:
        logger.info("auto-backbone: trying %s %s", cand_name, cand_kwargs or "")
        try:
            bb = get_backbone(
                cand_name, embed_dim=embed_dim, device=device, **cand_kwargs
            )
        except Exception as exc:  # get_backbone already guards, but be safe
            logger.warning("  %s raised %s; continuing", cand_name, exc)
            continue
        cls = type(bb).__name__
        last = bb
        if cls != "FallbackBackbone":
            logger.info("auto-backbone: using %s (%s)", cand_name, cls)
            info["auto_selected"] = cand_name
            info["auto_kwargs"] = cand_kwargs
            return bb, _finalize_backbone_info(info, bb)
        logger.info("  %s fell back to numpy; trying next candidate", cand_name)

    # Nothing real loaded → use the (already-built) fallback.
    logger.warning(
        "auto-backbone: no real backbone could be loaded; using numpy fallback."
    )
    if last is None:
        last = get_backbone("fallback", embed_dim=embed_dim, device=device)
    info["auto_selected"] = "fallback"
    return last, _finalize_backbone_info(info, last)


def _finalize_backbone_info(info: dict[str, Any], bb: Any) -> dict[str, Any]:
    cls = type(bb).__name__
    info["class"] = cls
    info["name"] = getattr(bb, "name", cls)
    info["embed_dim"] = int(getattr(bb, "embed_dim", 0))
    info["is_fallback"] = cls == "FallbackBackbone"
    info["multispectral_aware"] = cls in _MULTISPECTRAL_AWARE_CLASSES
    return info


# ---------------------------------------------------------------------------
# The orchestration
# ---------------------------------------------------------------------------
def run_real_proof(
    *,
    dataset: str = "eurosat",
    data_root: str = "./data",
    backbone: str = "auto",
    device: str = "auto",
    subset: int = 2000,
    gallery_per_class: int = 10,
    query_per_class: int = 5,
    train: bool = False,
    epochs: int = 15,
    ks: tuple[int, ...] = (5, 10),
    out: str = "./artifacts",
    report: Optional[str] = "docs/REAL_PROOF_RESULTS.md",
    seed: int = 0,
    cross_modal: bool = False,
    embed_dim: int = 256,
    latency_runs: int = 200,
    latency_warmup: int = 20,
    write_report: bool = True,
    samples: Optional[list] = None,
) -> dict[str, Any]:
    """Run the full real-data proof and return a results ``dict``.

    All knobs map 1:1 to the CLI flags. ``samples`` lets the (network-free) test
    inject a pre-built sample list so the orchestration can be exercised end to
    end without any download. The returned dict contains ``results`` (the
    :func:`~xsretrieval.eval.benchmark.evaluate` output), ``headline``, ``meta``,
    ``backbone``, ``env`` and ``timings``.
    """
    from xsretrieval.alignment.whitening import PerModalityWhitener
    from xsretrieval.eval.benchmark import evaluate, format_report
    from xsretrieval.retrieval.engine import RetrievalEngine

    t_start = time.time()
    timings: dict[str, float] = {}
    dev = resolve_device(device)
    env = env_report()
    logger.info("device=%s  env=%s", dev, _short_env(env))

    # 1) Data ---------------------------------------------------------------
    want_ms = bool(cross_modal) and dataset.lower() == "eurosat"
    data_info: dict[str, Any]
    if samples is None:
        if dataset.lower() != "eurosat":
            raise NotImplementedError(
                f"run_real_proof currently automates EuroSAT download/load; for "
                f"{dataset!r} prepare the data per docs/GPU_RUNBOOK.md and pass "
                f"--data-root, or use the package CLI. (sen12ms scanning is "
                f"supported by the adapters but the multi-hundred-GB download is "
                f"not automated here.)"
            )
        with _Step("download + load real EuroSAT") as st:
            samples, data_info = load_eurosat_samples(
                data_root, want_ms, subset, seed
            )
        timings["data"] = st.secs
    else:
        data_info = {"variant": "injected", "n_loaded": len(samples)}

    # 2) Split --------------------------------------------------------------
    with _Step("class-balanced query/gallery split") as st:
        queries, gallery = class_balanced_split(
            samples, gallery_per_class, query_per_class, seed
        )
    timings["split"] = st.secs
    if not queries or not gallery:
        raise RuntimeError(
            "empty query/gallery split — increase --subset or lower "
            "--gallery-per-class/--query-per-class"
        )

    # 3) Backbone -----------------------------------------------------------
    with _Step(f"build backbone ({backbone})") as st:
        bb, bb_info = build_backbone(backbone, embed_dim=embed_dim, device=dev)
    timings["backbone_build"] = st.secs
    logger.info(
        "backbone=%s (class=%s, fallback=%s, ms_aware=%s)",
        bb_info.get("name"), bb_info["class"], bb_info["is_fallback"],
        bb_info["multispectral_aware"],
    )

    # 4) Engine + whitener fit ---------------------------------------------
    whitener = PerModalityWhitener(shrinkage=0.9)
    engine = RetrievalEngine(
        bb,
        whitener=whitener,
        index_cfg={"index_type": "auto", "metric": "ip"},
        rerank=False,
    )
    with _Step("fit per-modality whitener on gallery") as st:
        engine.fit_whitener(gallery)
    timings["fit_whitener"] = st.secs

    # 5) Optional projection-head training ---------------------------------
    train_info: dict[str, Any] = {"enabled": bool(train)}
    if train:
        train_info.update(
            _train_projection_head(engine, bb, gallery, queries, dev, epochs, seed)
        )

    # 6) Encode + index + evaluate -----------------------------------------
    with _Step("encode gallery + build index + evaluate") as st:
        results = evaluate(
            engine,
            queries,
            gallery,
            ks=tuple(ks),
            recall_mode="raw",
            measure_latency=True,
            rerank=False,
            latency_warmup=latency_warmup,
            latency_runs=latency_runs,
        )
    timings["evaluate"] = st.secs

    backend = _index_backend(engine)
    meta = {
        "dataset": dataset,
        "data": data_info,
        "device": dev,
        "subset": subset,
        "gallery_per_class": gallery_per_class,
        "query_per_class": query_per_class,
        "n_queries": len(queries),
        "n_gallery": len(gallery),
        "index_backend": backend,
        "whitening": True,
        "seed": seed,
        "cross_modal_requested": bool(cross_modal),
    }
    results["meta"] = meta
    results["backbone"] = bb_info
    results["env"] = env
    results["timings"] = timings
    results["train"] = train_info

    # Cross-modal honesty gate. If MS was requested but the backbone is not
    # multispectral-aware, RGB≈MS so cross-modal numbers are not a real test.
    results["cross_modal_real"] = bool(
        cross_modal and bb_info["multispectral_aware"]
    )

    report_text = format_report(results)
    print("\n" + report_text + "\n")

    # 7) Save artifacts -----------------------------------------------------
    with _Step(f"save artifacts to {out}") as st:
        _save_artifacts(out, engine, results, meta, bb_info, train_info)
    timings["save"] = st.secs

    # 8) Write the Markdown report -----------------------------------------
    if write_report and report:
        _write_markdown_report(
            report, results, report_text,
            reproduction_cmd=_reproduction_command(
                dataset, backbone, dev, subset, gallery_per_class,
                query_per_class, train, cross_modal,
            ),
        )
        logger.info("wrote report to %s", report)

    timings["total"] = time.time() - t_start
    logger.info("DONE in %.1fs.  headline=%s", timings["total"],
                {k: round(v, 4) for k, v in results.get("headline", {}).items()})
    return results


def _train_projection_head(
    engine: Any, bb: Any, gallery: list, queries: list,
    dev: str, epochs: int, seed: int,
) -> dict[str, Any]:
    """Train the projection head on the gallery and wire it into *engine*.

    Returns a small info dict (final loss, val headline). Mutates *engine* so the
    subsequent evaluate() uses the trained head + a re-fitted whitener.
    """
    from xsretrieval.alignment.trainer import train_projection
    from xsretrieval.config import Config
    from xsretrieval.data.modalities import Modality

    cfg = Config.default()
    cfg.projection.enabled = True
    cfg.projection.out_dim = 256
    cfg.projection.hidden = 512
    cfg.backbone.embed_dim = int(getattr(bb, "embed_dim", 256))
    cfg.train.epochs = int(epochs)
    cfg.train.device = dev
    cfg.train.seed = int(seed)

    with _Step(f"train projection head ({epochs} epochs)") as st:
        result = train_projection(bb, gallery, cfg, val_samples=queries)
    head = result.head

    # Route the trained head through the engine (modality-aware projection).
    slot: dict[str, Any] = {"mod": None}

    class _RoutingBackbone:
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self.name = getattr(inner, "name", "backbone")
            self.embed_dim = getattr(inner, "embed_dim", 0)

        def embed(self, images: Any, modality: Any = None) -> np.ndarray:
            slot["mod"] = modality
            return self._inner.embed(images, modality)

    class _RoutingProjection:
        def forward(self, emb: np.ndarray) -> np.ndarray:
            import torch

            with torch.no_grad():
                t = torch.from_numpy(np.ascontiguousarray(emb, dtype=np.float32)).to(dev)
                out = head.forward(t, slot.get("mod"))
            return out.detach().cpu().numpy().astype(np.float32)

    engine.backbone = _RoutingBackbone(bb)
    engine.projection = _RoutingProjection()
    # Re-fit the whitener in the trained (projected) space.
    engine.fit_whitener(gallery)

    return {
        "enabled": True,
        "epochs": int(epochs),
        "device": dev,
        "final_loss": float(result.history[-1].get("total", float("nan")))
        if result.history else None,
        "val_headline": {k: float(v) for k, v in (result.val_metrics or {}).items()},
        "secs": st.secs,
    }


# ---------------------------------------------------------------------------
# Artifacts + report
# ---------------------------------------------------------------------------
def _index_backend(engine: Any) -> str:
    idx = engine.get_index()
    if idx is None:
        return "none"
    uses_faiss = getattr(idx, "uses_faiss", None)
    if isinstance(uses_faiss, bool):
        return "faiss" if uses_faiss else "numpy"
    return "unknown"


def _save_artifacts(
    out_dir: str,
    engine: Any,
    results: dict[str, Any],
    meta: dict[str, Any],
    bb_info: dict[str, Any],
    train_info: dict[str, Any],
) -> None:
    """Persist whitener, faiss/numpy index, used-config JSON, and results.json."""
    os.makedirs(out_dir, exist_ok=True)
    # Index.
    idx = engine.get_index()
    if idx is not None:
        try:
            idx.save(os.path.join(out_dir, "index"))
        except Exception as exc:  # pragma: no cover - backend dependent
            logger.warning("could not save index: %s", exc)
    # Whitener.
    if engine.whitener is not None and getattr(engine.whitener, "fitted_", False):
        try:
            engine.whitener.save(os.path.join(out_dir, "whitener.npz"))
        except Exception as exc:  # pragma: no cover
            logger.warning("could not save whitener: %s", exc)
    # Used args / config.
    used = {"meta": meta, "backbone": bb_info, "train": train_info,
            "env": results.get("env", {})}
    with open(os.path.join(out_dir, "used_args.json"), "w", encoding="utf-8") as fh:
        json.dump(_jsonable(used), fh, indent=2)
    # results.json (metrics + latency).
    payload = {
        "headline": results.get("headline", {}),
        "same_modal": results.get("same_modal", {}),
        "cross_modal": results.get("cross_modal", {}),
        "cells": results.get("cells", {}),
        "latency": results.get("latency", {}),
        "avg_query_time_ms": results.get("avg_query_time_ms"),
        "meta": meta,
        "backbone": bb_info,
        "train": train_info,
        "cross_modal_real": results.get("cross_modal_real", False),
        "config": results.get("config", {}),
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    logger.info("artifacts: index, whitener.npz, used_args.json, results.json")


def _write_markdown_report(
    path: str,
    results: dict[str, Any],
    report_text: str,
    reproduction_cmd: str,
) -> None:
    """Write the human-facing Markdown report with genuine numbers + caveats."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    meta = results.get("meta", {})
    bb = results.get("backbone", {})
    env = results.get("env", {})
    head = results.get("headline", {})
    lat = results.get("latency", {}) or {}
    train_info = results.get("train", {})
    data = meta.get("data", {})
    cm_real = results.get("cross_modal_real", False)
    cm_requested = meta.get("cross_modal_requested", False)

    md: list[str] = []
    md.append("# Real-Data Proof Results — `xsretrieval` (BAH 2026 PS-11)\n")
    md.append(
        "> Generated by `scripts/run_real_proof.py`. These are **genuine** "
        "numbers from **real satellite imagery** embedded with a **real "
        "backbone** — not the synthetic mechanism-demo from `smoke-test`.\n"
    )
    md.append("## Run configuration\n")
    md.append("| field | value |")
    md.append("|---|---|")
    md.append(f"| dataset | `{meta.get('dataset')}` ({data.get('variant', '?')}) |")
    md.append(f"| data root | `{data.get('root', '?')}` |")
    md.append(
        f"| backbone (requested → actual) | `{bb.get('requested')}` → "
        f"**`{bb.get('name')}`** (`{bb.get('class')}`) |"
    )
    md.append(f"| real backbone loaded? | **{not bb.get('is_fallback', True)}** "
              f"(fallback={bb.get('is_fallback')}) |")
    md.append(f"| multispectral-aware backbone? | {bb.get('multispectral_aware')} |")
    md.append(f"| device | `{meta.get('device')}` |")
    md.append(f"| embed dim | {bb.get('embed_dim')} |")
    md.append(f"| whitening | per-modality ZCA, enabled |")
    md.append(
        f"| projection head trained? | {bool(train_info.get('enabled'))}"
        + (f" ({train_info.get('epochs')} epochs)" if train_info.get('enabled') else "")
        + " |"
    )
    md.append(f"| subset (cap) | {meta.get('subset')} |")
    md.append(f"| images loaded | {data.get('n_loaded', '?')} of "
              f"{data.get('n_total_on_disk', '?')} on disk |")
    md.append(f"| gallery / query per class | {meta.get('gallery_per_class')} / "
              f"{meta.get('query_per_class')} |")
    md.append(f"| queries / gallery (total) | {meta.get('n_queries')} / "
              f"{meta.get('n_gallery')} |")
    md.append(f"| index backend | {meta.get('index_backend')} |")
    md.append(f"| seed | {meta.get('seed')} |")
    md.append("")
    md.append(
        f"Environment: torch `{env.get('torch')}`, cuda available "
        f"`{env.get('cuda_available')}`, faiss `{env.get('faiss')}`, "
        f"transformers `{env.get('transformers')}`.\n"
    )

    md.append("## Headline F1 (the PS-11 numbers)\n")
    md.append("| metric | same-modal | cross-modal |")
    md.append("|---|---|---|")
    for k in results.get("config", {}).get("ks", [5, 10]):
        if k in (5, 10):
            s = head.get(f"F1@{k}_same")
            c = head.get(f"F1@{k}_cross")
            md.append(
                f"| **F1@{k}** | {_fmt_md(s)} | {_fmt_md(c, cm_real)} |"
            )
    md.append("")
    if lat:
        md.append(
            f"**Average retrieval time per query** (batch=1, search-only): "
            f"**{lat.get('avg_query_time_ms', float('nan')):.3f} ms** "
            f"(p50 {lat.get('p50_query_time_ms', float('nan')):.3f} ms, "
            f"p95 {lat.get('p95_query_time_ms', float('nan')):.3f} ms; "
            f"{int(lat.get('n_runs', 0))} runs).\n"
        )

    md.append("## Honest notes\n")
    if not bb.get("is_fallback", True):
        md.append(
            f"- **Same-modal optical F1 is a real result.** Real EuroSAT "
            f"Sentinel-2 imagery, embedded with the real `{bb.get('name')}` "
            f"backbone ({bb.get('embed_dim')}-d), per-modality ZCA whitening, "
            f"class-balanced gallery so `R_q ≈ {meta.get('gallery_per_class')}` "
            f"(the F1@K lever — see `docs/GPU_RUNBOOK.md` §3a).\n"
        )
    else:
        md.append(
            "- **The configured real backbone could not be loaded in this "
            "environment, so the numpy `FallbackBackbone` ran.** The numbers "
            "below are real-imagery + hand-crafted-descriptor results, not a "
            "foundation-model result. Re-run on a box where the HF weights "
            "download to get the foundation-backbone numbers.\n"
        )
    if cm_requested and not cm_real:
        md.append(
            "- **Cross-modal real proof is DEFERRED (not reported as a real "
            "number).** The loaded backbone is RGB-only, so multispectral is "
            "reduced to pseudo-RGB and MS≈RGB — a cross-modal score here would "
            "be trivially inflated and is *not* a genuine optical↔MS test. The "
            "real cross-modal proof needs a multispectral-capable backbone "
            "(DOFA/CROMA) and ideally SEN12MS (SAR+optical), run on GPU.\n"
        )
    elif not cm_requested:
        md.append(
            "- **Cross-modal is not exercised in this run** (single-modality "
            "EuroSAT-RGB). The same-modal optical number is the guaranteed real "
            "result. For the full same- **and** cross-modal proof toward "
            "F1 ≥ 0.8, run on GPU with `--dataset sen12ms --backbone dofa "
            "--device cuda --train` (see `docs/GPU_RUNBOOK.md`).\n"
        )
    else:
        md.append(
            "- **Cross-modal numbers above are a genuine optical↔multispectral "
            "test** (a multispectral-aware backbone consumed the 13-band input "
            "differently from RGB).\n"
        )
    md.append(
        "- The headline uses `recall_mode: raw` (the grader's formula) and a "
        "leave-one-out protocol. F1@K is bounded by `2·r_K/(K+R_q)`; the "
        "class-balanced gallery sizes `R_q` toward `K` so a strong model can "
        "approach 0.8–1.0 (see `docs/GPU_RUNBOOK.md` §3a).\n"
    )
    md.append("## Reproduce\n")
    md.append("```bash\n" + reproduction_cmd + "\n```\n")
    md.append("## Full evaluation report (verbatim)\n")
    md.append("```\n" + report_text + "\n```\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))


def _reproduction_command(
    dataset: str, backbone: str, device: str, subset: int,
    gallery_per_class: int, query_per_class: int, train: bool, cross_modal: bool,
) -> str:
    parts = [
        "python scripts/run_real_proof.py",
        f"--dataset {dataset}",
        f"--backbone {backbone}",
        f"--device {device}",
        f"--subset {subset}",
        f"--gallery-per-class {gallery_per_class}",
        f"--query-per-class {query_per_class}",
        "--train" if train else "--no-train",
    ]
    if cross_modal:
        parts.append("--cross-modal")
    return " \\\n    ".join(parts)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def _short_env(env: dict[str, Any]) -> str:
    return (
        f"torch={env.get('torch')} cuda={env.get('cuda_available')} "
        f"faiss={env.get('faiss')} transformers={env.get('transformers')}"
    )


def _fmt_md(x: Optional[float], real: bool = True) -> str:
    if x is None:
        return "n/a"
    if not real:
        return f"_{x:.4f} (deferred — not a real cross-modal test)_"
    return f"**{x:.4f}**"


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse CLI for the real-data proof."""
    p = argparse.ArgumentParser(
        prog="run_real_proof",
        description=(
            "Automated, one-command real-data proof for xsretrieval: download "
            "real satellite imagery, embed with a real backbone, and produce "
            "genuine F1@5/@10 + latency."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", choices=["eurosat", "sen12ms"], default="eurosat")
    p.add_argument("--data-root", default="./data", help="dataset root directory")
    p.add_argument(
        "--backbone", default="auto",
        help="backbone name or 'auto' (try a real RS/vision backbone, fall back "
             "to numpy FallbackBackbone if weights cannot load)",
    )
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument(
        "--subset", type=int, default=2000,
        help="cap total images (class-balanced); <=0 means all",
    )
    p.add_argument("--gallery-per-class", type=int, default=10,
                   help="gallery items per class (≈ R_q, the F1@K lever)")
    p.add_argument("--query-per-class", type=int, default=5)
    train_grp = p.add_mutually_exclusive_group()
    train_grp.add_argument("--train", dest="train", action="store_true",
                           help="train the projection head before evaluating")
    train_grp.add_argument("--no-train", dest="train", action="store_false",
                           help="zero-shot (whitening only) [default]")
    p.set_defaults(train=False)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10])
    p.add_argument("--out", default="./artifacts", help="artifacts output dir")
    p.add_argument("--report", default="docs/REAL_PROOF_RESULTS.md",
                   help="Markdown report path ('' to skip)")
    p.add_argument(
        "--cross-modal", action="store_true",
        help="also attempt optical↔multispectral (EuroSAT all-bands). Only "
             "reported as a real result if a multispectral-aware backbone loads.",
    )
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--latency-runs", type=int, default=200)
    p.add_argument("--latency-warmup", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true", help="reduce log verbosity")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _seed_everything(args.seed)
    try:
        results = run_real_proof(
            dataset=args.dataset,
            data_root=args.data_root,
            backbone=args.backbone,
            device=args.device,
            subset=args.subset,
            gallery_per_class=args.gallery_per_class,
            query_per_class=args.query_per_class,
            train=args.train,
            epochs=args.epochs,
            ks=tuple(args.ks),
            out=args.out,
            report=(args.report or None),
            seed=args.seed,
            cross_modal=args.cross_modal,
            embed_dim=args.embed_dim,
            latency_runs=args.latency_runs,
            latency_warmup=args.latency_warmup,
        )
    except Exception as exc:
        logger.error("real proof failed: %s", exc, exc_info=not args.quiet)
        return 1
    # Non-zero only on hard failure; a fallback backbone still "succeeds" (and
    # is clearly recorded), so CI / automation can detect availability.
    return 0 if results.get("headline") else 2


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
