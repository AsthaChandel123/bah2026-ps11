"""xsretrieval — cross-modal satellite image retrieval (BAH 2026 PS-11).

A CPU-first toolkit for retrieving semantically-matching satellite imagery
*across* sensor modalities (optical RGB, multispectral, SAR, …). The pipeline is

    images ─► backbone.embed ─► projection ─► per-modality whitening ─► L2-norm
            ─► FAISS / numpy shared index ─► (optional k-reciprocal re-rank) ─► top-k

and is scored by F1@5 / F1@10 for same-modal and cross-modal retrieval plus the
average per-query latency.

Top-level convenience exports
-----------------------------
* :class:`~xsretrieval.data.modalities.Modality`, :class:`~xsretrieval.data.modalities.Sample`
* :func:`~xsretrieval.models.get_backbone`
* :class:`~xsretrieval.retrieval.engine.RetrievalEngine`
* :class:`~xsretrieval.alignment.whitening.PerModalityWhitener`
* :class:`~xsretrieval.index.faiss_index.RetrievalIndex`
* :func:`~xsretrieval.eval.benchmark.evaluate`
* :func:`~xsretrieval.data.synthetic.make_synthetic_multimodal`
* :func:`~xsretrieval.pipeline.build_pipeline`, :func:`~xsretrieval.config.Config`

Lazy imports
------------
``import xsretrieval`` is deliberately cheap: nothing heavy (torch / faiss /
transformers / timm / rasterio) is imported at package-load time. The public
names above are resolved on first access via :pep:`562` module ``__getattr__``,
each pulling only its own light submodule (numpy-only). So a bare-numpy install
can ``import xsretrieval`` and reach :class:`Modality` / :class:`RetrievalEngine`
/ :func:`evaluate` without the optional extras present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

# Map an exported name -> (submodule, attribute) for lazy resolution (PEP 562).
_EXPORTS: dict[str, tuple[str, str]] = {
    "Modality": ("xsretrieval.data.modalities", "Modality"),
    "Sample": ("xsretrieval.data.modalities", "Sample"),
    "MODALITY_CHANNELS": ("xsretrieval.data.modalities", "MODALITY_CHANNELS"),
    "get_backbone": ("xsretrieval.models", "get_backbone"),
    "RetrievalEngine": ("xsretrieval.retrieval.engine", "RetrievalEngine"),
    "PerModalityWhitener": ("xsretrieval.alignment.whitening", "PerModalityWhitener"),
    "RetrievalIndex": ("xsretrieval.index.faiss_index", "RetrievalIndex"),
    "evaluate": ("xsretrieval.eval.benchmark", "evaluate"),
    "format_report": ("xsretrieval.eval.benchmark", "format_report"),
    "make_synthetic_multimodal": (
        "xsretrieval.data.synthetic",
        "make_synthetic_multimodal",
    ),
    "make_synthetic_embeddings": (
        "xsretrieval.data.synthetic",
        "make_synthetic_embeddings",
    ),
    "Config": ("xsretrieval.config", "Config"),
    "build_pipeline": ("xsretrieval.pipeline", "build_pipeline"),
    "run_evaluation": ("xsretrieval.pipeline", "run_evaluation"),
}

__all__ = ["__version__", *sorted(_EXPORTS)]

if TYPE_CHECKING:  # pragma: no cover - import for type checkers / IDEs only
    from xsretrieval.alignment.whitening import PerModalityWhitener
    from xsretrieval.config import Config
    from xsretrieval.data.modalities import MODALITY_CHANNELS, Modality, Sample
    from xsretrieval.data.synthetic import (
        make_synthetic_embeddings,
        make_synthetic_multimodal,
    )
    from xsretrieval.eval.benchmark import evaluate, format_report
    from xsretrieval.index.faiss_index import RetrievalIndex
    from xsretrieval.models import get_backbone
    from xsretrieval.pipeline import build_pipeline, run_evaluation
    from xsretrieval.retrieval.engine import RetrievalEngine


def __getattr__(name: str) -> Any:  # PEP 562: lazy, cheap top-level attributes
    """Resolve a top-level export on first access (keeps ``import`` fast)."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    return getattr(module, target[1])


def __dir__() -> list[str]:  # pragma: no cover - cosmetic
    return sorted(set(__all__) | set(globals()))
