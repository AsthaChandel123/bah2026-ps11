"""High-level pipeline orchestration for ``xsretrieval``.

Turns a :class:`~xsretrieval.config.Config` into a runnable
:class:`~xsretrieval.retrieval.engine.RetrievalEngine` and provides the
end-to-end orchestrators the CLI / API / demo call:

* :func:`build_pipeline`        — config → engine (backbone + optional projection
  head + optional per-modality whitener).
* :func:`load_samples`          — load the configured dataset, transparently
  falling back to synthetic data when the real dataset is absent.
* :func:`run_evaluation`        — load → query/gallery split → **fit whitener on
  the gallery** → index → evaluate → structured results dict.
* :func:`build_index_from_dataset` — build + return an indexed engine (for
  serving / saving).
* :func:`encode_dataset`        — encode all samples to embeddings (+ metadata).

The default config enables whitening, so the headline cross-modal F1 reflects
the modality-gap fix. Heavy dependencies stay lazy (only numpy is needed for the
fallback path); torch/faiss are pulled in only if the chosen backbone / index
actually needs them.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from xsretrieval.config import Config
from xsretrieval.data.modalities import Modality, Sample

logger = logging.getLogger(__name__)

__all__ = [
    "build_pipeline",
    "load_samples",
    "make_query_gallery",
    "run_evaluation",
    "build_index_from_dataset",
    "encode_dataset",
]


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------
def _build_whitener(cfg: Config):
    """Construct an unfitted :class:`PerModalityWhitener` from the config."""
    from xsretrieval.alignment.whitening import PerModalityWhitener

    wc = cfg.whitening
    return PerModalityWhitener(
        n_components=wc.n_components,
        remove_top_pc=wc.remove_top_pc,
        shrinkage=wc.shrinkage,
    )


def _build_projection(cfg: Config):
    """Construct (and optionally load) the projection head, or ``None``.

    Requires torch only when ``projection.enabled`` is ``True``.
    """
    pc = cfg.projection
    if not pc.enabled:
        return None
    from xsretrieval.models.projection import ProjectionHeads

    head = ProjectionHeads(
        in_dim=cfg.backbone.embed_dim,
        out_dim=pc.out_dim,
        modalities=[Modality(m) for m in cfg.data.modalities],
        share_final=pc.share_final,
        hidden=pc.hidden,
        dropout=pc.dropout,
    )
    if pc.weights:
        import torch  # local, lazy

        state = torch.load(pc.weights, map_location="cpu")
        head.load_state_dict(state)
        logger.info("Loaded projection weights from %s", pc.weights)
    head.eval()
    return _TorchProjectionAdapter(head, [Modality(m) for m in cfg.data.modalities])


class _TorchProjectionAdapter:
    """Adapt a torch ``ProjectionHeads`` to the engine's ``forward(emb)`` contract.

    The engine encodes one modality at a time and calls ``projection.forward(emb)``
    with a numpy ``(B, D)`` batch (no modality argument). This adapter remembers
    the modality of the *current* encode call so the shared head can route to the
    right per-modality first layer. Because the engine groups by modality before
    encoding, setting the modality just before each batch is sufficient.
    """

    def __init__(self, head: Any, modalities: list[Modality]) -> None:
        self.head = head
        self._modalities = modalities
        self._current: Optional[Modality] = None

    def set_modality(self, modality: Modality) -> None:
        self._current = modality

    def forward(self, emb: np.ndarray) -> np.ndarray:
        import torch  # local, lazy

        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(emb, dtype=np.float32))
            out = self.head.forward(t, self._current)
        return out.detach().cpu().numpy().astype(np.float32)


def build_pipeline(config: Config) -> "Any":
    """Build a :class:`RetrievalEngine` from *config* (no data loaded yet).

    Constructs the backbone via :func:`~xsretrieval.models.get_backbone` (with
    graceful fallback), an optional projection head, and an optional
    (still-unfitted) per-modality whitener. The returned engine must be given a
    gallery — via :meth:`RetrievalEngine.index_gallery` or :func:`run_evaluation`
    — before it can answer queries; if whitening is enabled, fit the whitener on
    the gallery first (``engine.fit_whitener(gallery)``).

    Parameters
    ----------
    config:
        The pipeline configuration.

    Returns
    -------
    RetrievalEngine
        A configured engine (backbone + projection + whitener wired in).
    """
    from xsretrieval.models import get_backbone
    from xsretrieval.retrieval.engine import RetrievalEngine

    bc = config.backbone
    backbone = get_backbone(bc.name, embed_dim=bc.embed_dim, **dict(bc.kwargs))
    projection = _build_projection(config)
    whitener = _build_whitener(config) if config.whitening.enabled else None

    index_cfg = {
        "index_type": config.index.type,
        "metric": config.index.metric,
        "nprobe": config.index.nprobe,
    }
    engine = RetrievalEngine(
        backbone,
        projection=projection,
        whitener=whitener,
        index_cfg=index_cfg,
        rerank=config.index.rerank,
    )
    return engine


# ---------------------------------------------------------------------------
# Data loading (with synthetic fallback)
# ---------------------------------------------------------------------------
def _synthetic_samples(cfg: Config) -> list[Sample]:
    """Generate a synthetic multi-modal dataset matching the data config.

    Two substrates:

    * ``"embedding"`` — pre-encoded vectors with an explicit modality gap, packed
      as ``(D, 1, 1)`` "images" so the ``precomputed`` backbone round-trips them.
      This is the regime where per-modality whitening demonstrably lifts
      cross-modal F1.
    * ``"image"`` — rendered multi-modal images (exercises the full encode path).
    """
    dc = cfg.data
    modalities = [Modality(m) for m in dc.modalities]

    if dc.substrate == "embedding":
        from xsretrieval.data.synthetic import make_synthetic_embeddings

        dim = cfg.backbone.embed_dim
        emb, labels, mod_codes = make_synthetic_embeddings(
            n_classes=dc.n_classes,
            per_class_per_modality=dc.per_class,
            modalities=modalities,
            dim=dim,
            seed=dc.seed,
            modality_shift=dc.modality_shift,
        )
        samples: list[Sample] = []
        for i in range(emb.shape[0]):
            m = modalities[int(mod_codes[i])]
            loc = f"cls{int(labels[i])}_loc{i}"
            samples.append(
                Sample(
                    id=f"{m.value}_{i}",
                    image=emb[i].astype(np.float32).reshape(dim, 1, 1),
                    modality=m,
                    label=int(labels[i]),
                    location_id=loc,
                    meta={"synthetic": True, "substrate": "embedding"},
                )
            )
        return samples

    from xsretrieval.data.synthetic import make_synthetic_multimodal

    return make_synthetic_multimodal(
        n_classes=dc.n_classes,
        per_class_per_modality=dc.per_class,
        modalities=modalities,
        size=dc.size,
        seed=dc.seed,
    )


def load_samples(config: Config) -> list[Sample]:
    """Load the configured dataset as a flat ``list[Sample]``.

    For ``dataset == "synthetic"`` (or when a real dataset cannot be found on
    disk) a synthetic multi-modal dataset is generated so a run always
    completes. Real adapters (``eurosat`` / ``sen12ms`` / ``folder``) are loaded
    lazily and, on any :class:`FileNotFoundError`, fall back to synthetic with a
    warning.

    Returns
    -------
    list[Sample]
        The samples spanning all configured modalities.
    """
    dc = config.data
    if dc.dataset.lower() == "synthetic":
        return _synthetic_samples(config)

    try:
        from xsretrieval.data.datasets import get_dataset

        kwargs: dict[str, Any] = {}
        if dc.root is not None:
            kwargs["root"] = dc.root
        adapter = get_dataset(dc.dataset, **kwargs)
        samples = _materialize(adapter)
        if not samples:
            raise FileNotFoundError(f"dataset {dc.dataset!r} produced no samples")
        logger.info("Loaded %d samples from dataset %r", len(samples), dc.dataset)
        return samples
    except (FileNotFoundError, KeyError, ImportError) as exc:
        logger.warning(
            "Could not load dataset %r (%s); falling back to synthetic data.",
            dc.dataset,
            exc,
        )
        return _synthetic_samples(config)


def _materialize(adapter: Any) -> list[Sample]:
    """Best-effort extraction of a flat ``list[Sample]`` from a dataset adapter."""
    if hasattr(adapter, "samples"):
        return list(adapter.samples)
    if hasattr(adapter, "__iter__"):
        return list(adapter)
    if hasattr(adapter, "__len__") and hasattr(adapter, "__getitem__"):
        return [adapter[i] for i in range(len(adapter))]
    raise FileNotFoundError("dataset adapter exposes no samples")


def make_query_gallery(
    config: Config, samples: list[Sample]
) -> tuple[list[Sample], list[Sample]]:
    """Split *samples* into (queries, gallery), per-modality and class-stratified.

    Each modality is split independently so **every** modality is represented in
    both the query and gallery sets — a prerequisite for filling the full
    query×gallery evaluation matrix (same-modal diagonal + cross-modal
    off-diagonal). Deterministic given ``config.data.seed``.
    """
    from xsretrieval.data.datasets import make_query_gallery_split

    dc = config.data
    by_mod: dict[Modality, list[Sample]] = defaultdict(list)
    for s in samples:
        by_mod[s.modality if isinstance(s.modality, Modality) else Modality(s.modality)].append(s)

    queries: list[Sample] = []
    gallery: list[Sample] = []
    for items in by_mod.values():
        q, g = make_query_gallery_split(
            items,
            query_frac=dc.query_frac,
            class_balanced_gallery=dc.class_balanced_gallery,
            seed=dc.seed,
        )
        queries.extend(q)
        gallery.extend(g)
    return queries, gallery


# ---------------------------------------------------------------------------
# Whitener fitting
# ---------------------------------------------------------------------------
def _fit_whitener_if_enabled(
    engine: Any, config: Config, gallery: list[Sample], queries: list[Sample]
) -> None:
    """Fit the engine's whitener on the configured reference set (in place)."""
    if engine.whitener is None:
        return
    fit_on = config.whitening.fit_on
    reference = gallery if fit_on != "all" else (list(gallery) + list(queries))
    engine.fit_whitener(reference)
    logger.info(
        "Fitted per-modality whitener on %d reference samples (fit_on=%r).",
        len(reference),
        fit_on,
    )


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------
def run_evaluation(
    config: Config,
    samples: Optional[list[Sample]] = None,
) -> dict[str, Any]:
    """Run the full PS-11 evaluation for *config* and return the results dict.

    Steps: load data (or use *samples*) → per-modality query/gallery split → fit
    the whitener on the gallery (if enabled) → index the gallery → evaluate the
    query×gallery matrix (P/R/F1/nDCG@k + mAP per cell, same/cross aggregates,
    four headline F1s, latency).

    Parameters
    ----------
    config:
        Pipeline configuration (whitening enabled by default).
    samples:
        Optional pre-loaded samples; if ``None``, :func:`load_samples` is used
        (with synthetic fallback).

    Returns
    -------
    dict
        The :func:`~xsretrieval.eval.benchmark.evaluate` result, augmented with a
        ``"meta"`` block (backbone class actually built, whitening flag, dataset,
        substrate, sample counts).
    """
    from xsretrieval.eval.benchmark import evaluate

    if samples is None:
        samples = load_samples(config)
    queries, gallery = make_query_gallery(config, samples)
    if not queries or not gallery:
        raise RuntimeError(
            "query/gallery split is empty; check the dataset and query_frac"
        )

    engine = build_pipeline(config)
    _fit_whitener_if_enabled(engine, config, gallery, queries)

    ec = config.eval
    results = evaluate(
        engine,
        queries,
        gallery,
        ks=tuple(ec.ks),
        recall_mode=ec.recall_mode,
        measure_latency=ec.measure_latency,
        rerank=config.index.rerank,
        latency_warmup=ec.latency_warmup,
        latency_runs=ec.latency_runs,
    )
    results["meta"] = {
        "config_name": config.name,
        "backbone": config.backbone.name,
        "backbone_class": type(engine.backbone).__name__,
        "whitening": bool(engine.whitener is not None),
        "projection": bool(engine.projection is not None),
        "dataset": config.data.dataset,
        "substrate": config.data.substrate,
        "n_samples": len(samples),
        "n_queries": len(queries),
        "n_gallery": len(gallery),
        "index_backend": _index_backend(engine),
    }
    return results


def _index_backend(engine: Any) -> str:
    """Report whether the active index used faiss or the numpy fallback."""
    idx = engine.get_index()
    if idx is None:
        return "none"
    uses_faiss = getattr(idx, "uses_faiss", None)
    if isinstance(uses_faiss, bool):
        return "faiss" if uses_faiss else "numpy"
    return "unknown"


def build_index_from_dataset(
    config: Config, samples: Optional[list[Sample]] = None
) -> tuple["Any", dict[str, Any]]:
    """Build and return an indexed engine (gallery indexed, whitener fitted).

    Useful for serving (API / demo) and for persisting an index. Uses the
    **whole dataset** as the gallery (no held-out queries) so the served archive
    is complete.

    Returns
    -------
    (engine, info):
        The indexed :class:`RetrievalEngine` and a small info dict.
    """
    if samples is None:
        samples = load_samples(config)
    engine = build_pipeline(config)
    _fit_whitener_if_enabled(engine, config, samples, [])
    engine.index_gallery(samples)
    info = {
        "n_gallery": len(samples),
        "embed_dim": engine.embed_dim,
        "backbone_class": type(engine.backbone).__name__,
        "whitening": bool(engine.whitener is not None),
        "index_backend": _index_backend(engine),
    }
    return engine, info


def encode_dataset(
    config: Config, samples: Optional[list[Sample]] = None
) -> dict[str, Any]:
    """Encode every sample to a (whitened) embedding and return arrays + metadata.

    Returns
    -------
    dict with ``embeddings`` ``(N, D)``, ``labels`` ``(N,)``, ``modalities``
    ``(N,)`` (string values), ``ids`` ``(N,)`` and ``location_ids`` ``(N,)``.
    """
    if samples is None:
        samples = load_samples(config)
    engine = build_pipeline(config)
    _fit_whitener_if_enabled(engine, config, samples, [])
    embeddings = engine.encode(samples)
    return {
        "embeddings": embeddings,
        "labels": np.array([int(s.label) for s in samples], dtype=np.int64),
        "modalities": np.array(
            [(s.modality.value if isinstance(s.modality, Modality) else str(s.modality)) for s in samples],
            dtype=object,
        ),
        "ids": np.array([s.id for s in samples], dtype=object),
        "location_ids": np.array(
            [("" if s.location_id is None else str(s.location_id)) for s in samples],
            dtype=object,
        ),
    }
