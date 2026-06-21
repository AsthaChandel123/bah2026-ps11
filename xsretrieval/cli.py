"""Command-line interface for ``xsretrieval``.

Run as ``python -m xsretrieval.cli <command>`` (or, once installed, the
``xsretrieval`` console script). Subcommands:

* ``smoke-test`` — end-to-end synthetic pipeline **with whitening**; prints the
  query×gallery matrix, the four headline F1@5/@10 (same + cross), latency, and a
  whitening on/off comparison. Exits 0 on success.
* ``evaluate``   — full evaluation on a configured dataset (synthetic fallback).
* ``build-index``— build + persist a retrieval bundle (index + whitener + config).
* ``query``      — query a persisted bundle with an image, print top-k results.
* ``train``      — train a projection head (torch) and report val metrics.
* ``info``       — print environment / backbone availability.

Heavy imports (torch / faiss / the pipeline) live **inside** the subcommand
handlers, so ``python -m xsretrieval.cli info`` is fast and works on bare numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

__all__ = ["main", "build_parser"]


# ---------------------------------------------------------------------------
# Config loading helper
# ---------------------------------------------------------------------------
def _load_config(path: Optional[str]):
    """Load a :class:`Config` from YAML, or the built-in default."""
    from xsretrieval.config import Config

    if path:
        return Config.from_yaml(path)
    return Config.default()


# ---------------------------------------------------------------------------
# Persisted bundle (index + whitener + config + gallery meta)
# ---------------------------------------------------------------------------
def _save_bundle(out_dir: str, engine: Any, config: Any) -> None:
    """Persist an indexed engine + whitener + config + gallery metadata."""
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    index = engine.get_index()
    if index is None:
        raise RuntimeError("engine has no index to save; build it first")
    index.save(os.path.join(out_dir, "index"))
    if engine.whitener is not None and getattr(engine.whitener, "fitted_", False):
        engine.whitener.save(os.path.join(out_dir, "whitener.npz"))
    # Extra gallery metadata the index does not persist (location ids).
    np.savez(
        os.path.join(out_dir, "gallery_meta.npz"),
        location_ids=np.asarray(
            engine.gallery_location_ids
            if engine.gallery_location_ids is not None
            else [],
            dtype=object,
        ),
        embed_dim=np.array([engine.embed_dim or 0], dtype=np.int64),
    )
    config.to_yaml(os.path.join(out_dir, "config.yaml"))


def _load_bundle(in_dir: str):
    """Reconstruct an engine for serving from a persisted bundle directory."""
    import numpy as np

    from xsretrieval.alignment.whitening import PerModalityWhitener
    from xsretrieval.index.faiss_index import RetrievalIndex
    from xsretrieval.models import get_backbone
    from xsretrieval.retrieval.engine import RetrievalEngine

    config = _load_config(os.path.join(in_dir, "config.yaml"))
    backbone = get_backbone(
        config.backbone.name,
        embed_dim=config.backbone.embed_dim,
        **dict(config.backbone.kwargs),
    )
    whitener = None
    wpath = os.path.join(in_dir, "whitener.npz")
    if os.path.exists(wpath):
        whitener = PerModalityWhitener.load(wpath)

    engine = RetrievalEngine(backbone, whitener=whitener, rerank=config.index.rerank)
    index = RetrievalIndex.load(os.path.join(in_dir, "index"))
    engine.index = index
    engine.gallery_ids = index.ids
    engine.gallery_labels = index.labels
    engine.gallery_modalities = index.modalities
    meta = np.load(os.path.join(in_dir, "gallery_meta.npz"), allow_pickle=True)
    locs = np.asarray(meta["location_ids"], dtype=object)
    engine.gallery_location_ids = locs if locs.size else None
    engine.embed_dim = int(meta["embed_dim"][0]) or index.dim
    return engine, config


# ---------------------------------------------------------------------------
# Subcommand: smoke-test
# ---------------------------------------------------------------------------
def cmd_smoke_test(args: argparse.Namespace) -> int:
    """Run the synthetic end-to-end pipeline with whitening and print a report."""
    from xsretrieval.config import Config
    from xsretrieval.eval.benchmark import format_report
    from xsretrieval.pipeline import run_evaluation

    if args.config:
        config = _load_config(args.config)
    else:
        # Default smoke config: embedding substrate (a real modality gap) so the
        # per-modality whitening win is demonstrated honestly, on a fast,
        # dependency-free numpy path.
        config = Config.default()
        config.name = "smoke-test"
        config.data.substrate = "embedding"
        config.data.dataset = "synthetic"
        config.data.n_classes = 10
        config.data.per_class = 20
        config.backbone.name = "precomputed"
        config.backbone.embed_dim = 128
        config.eval.latency_runs = 200

    print(f"[smoke-test] config={config.name} backbone={config.backbone.name} "
          f"substrate={config.data.substrate} whitening={config.whitening.enabled}")
    print("[smoke-test] running WITH per-modality whitening (default pipeline) ...\n")

    results = run_evaluation(config)
    print(format_report(results))

    meta = results.get("meta", {})
    print(
        f"\n[meta] backbone_class={meta.get('backbone_class')}  "
        f"index_backend={meta.get('index_backend')}  "
        f"queries={meta.get('n_queries')}  gallery={meta.get('n_gallery')}"
    )

    # Whitening on/off comparison (proves the modality-gap fix lifts cross-modal).
    cfg_off = _copy_config(config)
    cfg_off.whitening.enabled = False
    cfg_off.eval.measure_latency = False
    res_off = run_evaluation(cfg_off)
    h_on, h_off = results["headline"], res_off["headline"]
    print("\nWhitening ablation (headline F1):")
    print("  metric          whitening OFF    whitening ON")
    for key in ("F1@5_same", "F1@5_cross", "F1@10_same", "F1@10_cross"):
        if key in h_on:
            print(f"  {key:<14} {h_off.get(key, 0.0):>12.4f}    {h_on.get(key, 0.0):>12.4f}")

    cross_on = h_on.get("F1@10_cross", 0.0)
    cross_off = h_off.get("F1@10_cross", 0.0)
    # Chance F1@10 for the configured setup (rough upper bound on random retrieval).
    chance = _chance_f1(config)
    print(
        f"\n[check] cross-modal F1@10: whitening OFF={cross_off:.4f}  "
        f"ON={cross_on:.4f}  (random≈{chance:.4f})"
    )
    ok = cross_on > max(3.0 * chance, 0.10) and cross_on >= cross_off
    if ok:
        print("[smoke-test] PASS — whitened cross-modal F1 is clearly above chance "
              "and improved by whitening.")
        return 0
    print("[smoke-test] WARNING — cross-modal F1 not clearly above chance; "
          "check the configuration.")
    return 1


def _chance_f1(config: Any) -> float:
    """Rough random-retrieval F1@10 for the configured class count.

    With C balanced classes, the precision of random retrieval ≈ 1/C; recall@10
    is bounded similarly. We report the precision-side estimate (a conservative
    proxy for the random F1 ceiling) so the smoke-test "above chance" check has a
    concrete reference.
    """
    c = max(1, int(config.data.n_classes))
    return 1.0 / c


# ---------------------------------------------------------------------------
# Subcommand: evaluate
# ---------------------------------------------------------------------------
def cmd_evaluate(args: argparse.Namespace) -> int:
    """Full evaluation on the configured dataset (synthetic fallback)."""
    from xsretrieval.eval.benchmark import format_report
    from xsretrieval.pipeline import run_evaluation

    config = _load_config(args.config)
    if args.no_latency:
        config.eval.measure_latency = False
    results = run_evaluation(config)
    print(format_report(results))
    meta = results.get("meta", {})
    print(
        f"\n[meta] config={meta.get('config_name')}  "
        f"backbone={meta.get('backbone')} ({meta.get('backbone_class')})  "
        f"whitening={meta.get('whitening')}  index={meta.get('index_backend')}  "
        f"dataset={meta.get('dataset')}"
    )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(_jsonable(results), fh, indent=2)
        print(f"[evaluate] wrote JSON results to {args.json}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: build-index
# ---------------------------------------------------------------------------
def cmd_build_index(args: argparse.Namespace) -> int:
    """Build a retrieval bundle from a dataset and persist it to ``--out``."""
    from xsretrieval.pipeline import build_index_from_dataset

    config = _load_config(args.config)
    engine, info = build_index_from_dataset(config)
    _save_bundle(args.out, engine, config)
    print(f"[build-index] indexed {info['n_gallery']} gallery items "
          f"(dim={info['embed_dim']}, backend={info['index_backend']}, "
          f"whitening={info['whitening']})")
    print(f"[build-index] bundle saved to {args.out}/ "
          f"(index, whitener, config, gallery_meta)")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: query
# ---------------------------------------------------------------------------
def cmd_query(args: argparse.Namespace) -> int:
    """Query a persisted bundle with an image and print the top-k results."""
    import numpy as np

    from xsretrieval.data.modalities import Modality

    engine, config = _load_bundle(args.index)

    modality = Modality(args.modality) if args.modality else Modality(config.data.modalities[0])
    image = _load_query_image(args.image, config, modality)
    emb = engine.encode([_as_sample(image, modality)])
    hits = engine.query(
        emb[0] if emb.ndim == 2 else emb,
        k=args.k,
        gallery_modality=(Modality(args.gallery_modality) if args.gallery_modality else None),
        exclude_same_location=False,
    )
    print(f"[query] image={args.image} modality={modality.value} k={args.k} "
          f"gallery_modality={args.gallery_modality or 'all'}")
    print(f"{'rank':<5}{'id':<28}{'modality':<16}{'label':<8}{'score':>8}")
    for h in hits:
        print(f"{h['rank']:<5}{str(h['id']):<28}{str(h['modality']):<16}"
              f"{h['label']:<8}{h['score']:>8.4f}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(_jsonable(hits), fh, indent=2)
    return 0


def _load_query_image(path: str, config: Any, modality: Any):
    """Load a query image: ``.npy`` embedding/array, or an image file."""
    import numpy as np

    if path.endswith(".npy"):
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 1:  # an embedding vector -> (D, 1, 1) for precomputed
            arr = arr.reshape(arr.shape[0], 1, 1)
        return arr
    from xsretrieval.data.datasets import load_image_any

    arr = load_image_any(path).astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] <= 16 and arr.shape[0] > arr.shape[2]:
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return arr


def _as_sample(image, modality):
    """Wrap a raw image array in a minimal Sample for the engine's encode()."""
    from xsretrieval.data.modalities import Sample

    return Sample(id="__query__", image=image, modality=modality, label=-1)


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> int:
    """Train a projection head (torch) and optionally persist it."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("[train] PyTorch is not installed. Install it with "
              "`pip install torch` (see requirements-cpu.txt).", file=sys.stderr)
        return 2

    from xsretrieval.alignment.trainer import train_projection
    from xsretrieval.models import get_backbone
    from xsretrieval.pipeline import load_samples, make_query_gallery

    config = _load_config(args.config)
    config.projection.enabled = True
    if args.epochs:
        config.train.epochs = args.epochs

    samples = load_samples(config)
    queries, gallery = make_query_gallery(config, samples)
    # Use the (larger) gallery as training data and the held-out queries as
    # validation, so we train on more data and validate on unseen samples.
    train_samples, val_samples = gallery, queries
    backbone = get_backbone(
        config.backbone.name,
        embed_dim=config.backbone.embed_dim,
        **dict(config.backbone.kwargs),
    )
    print(f"[train] backbone={type(backbone).__name__} "
          f"train={len(train_samples)} val={len(val_samples)} "
          f"epochs={config.train.epochs}")
    result = train_projection(backbone, train_samples, config, val_samples=val_samples)
    print(f"[train] done. final loss={result.history[-1].get('total', float('nan')):.4f}")
    if result.val_metrics:
        print("[train] validation headline F1:")
        for k, v in result.val_metrics.items():
            print(f"  {k:<14} {v:.4f}")
    if args.out:
        torch.save(result.head.state_dict(), args.out)
        print(f"[train] saved projection head weights to {args.out}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: info
# ---------------------------------------------------------------------------
def cmd_info(args: argparse.Namespace) -> int:
    """Print environment + backbone availability."""
    import importlib

    import xsretrieval

    print(f"xsretrieval {xsretrieval.__version__}")
    print(f"python {sys.version.split()[0]}  platform {sys.platform}")
    print("\nOptional dependencies:")
    for mod in ("numpy", "torch", "faiss", "transformers", "timm", "open_clip",
                "huggingface_hub", "sklearn", "yaml", "rasterio", "PIL",
                "fastapi", "gradio"):
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  {mod:<16} OK ({ver})")
        except Exception:
            print(f"  {mod:<16} -- not installed")

    print("\nRegistered backbones:")
    from xsretrieval.models.backbones.base import REGISTRY, _import_concrete_backbones

    _import_concrete_backbones()
    for name in sorted(REGISTRY):
        print(f"  {name}")

    print("\nDefault pipeline:")
    from xsretrieval.config import Config

    c = Config.default()
    print(f"  backbone={c.backbone.name} (embed_dim={c.backbone.embed_dim})")
    print(f"  whitening={c.whitening.enabled} (shrinkage={c.whitening.shrinkage})")
    print(f"  index={c.index.type}/{c.index.metric}  ks={c.eval.ks}")
    return 0


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy / tuple types to JSON-serialisable Python."""
    import numpy as np

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


def _copy_config(config: Any):
    """Deep-copy a Config via its dict round-trip."""
    from xsretrieval.config import Config

    return Config.from_dict(config.to_dict())


# ---------------------------------------------------------------------------
# Parser + entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="xsretrieval",
        description="Cross-modal satellite image retrieval (BAH 2026 PS-11).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_smoke = sub.add_parser(
        "smoke-test", help="run the synthetic end-to-end pipeline with whitening"
    )
    p_smoke.add_argument("--config", default=None, help="optional config YAML")
    p_smoke.set_defaults(func=cmd_smoke_test)

    p_eval = sub.add_parser("evaluate", help="full evaluation on a configured dataset")
    p_eval.add_argument("--config", required=True, help="config YAML path")
    p_eval.add_argument("--json", default=None, help="write results JSON to this path")
    p_eval.add_argument("--no-latency", action="store_true", help="skip latency benchmark")
    p_eval.set_defaults(func=cmd_evaluate)

    p_build = sub.add_parser("build-index", help="build + persist a retrieval bundle")
    p_build.add_argument("--config", required=True, help="config YAML path")
    p_build.add_argument("--out", required=True, help="output bundle directory")
    p_build.set_defaults(func=cmd_build_index)

    p_query = sub.add_parser("query", help="query a persisted bundle with an image")
    p_query.add_argument("--index", required=True, help="bundle directory (from build-index)")
    p_query.add_argument("--image", required=True, help="query image (or .npy embedding)")
    p_query.add_argument("--k", type=int, default=10, help="number of results")
    p_query.add_argument("--modality", default=None, help="query modality")
    p_query.add_argument("--gallery-modality", default=None, help="restrict gallery modality")
    p_query.add_argument("--json", default=None, help="write hits JSON to this path")
    p_query.set_defaults(func=cmd_query)

    p_train = sub.add_parser("train", help="train a projection head (torch)")
    p_train.add_argument("--config", required=True, help="config YAML path")
    p_train.add_argument("--epochs", type=int, default=None, help="override epochs")
    p_train.add_argument("--out", default=None, help="save head weights to this path")
    p_train.set_defaults(func=cmd_train)

    p_info = sub.add_parser("info", help="print environment / backbone availability")
    p_info.set_defaults(func=cmd_info)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    import logging

    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
