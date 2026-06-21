"""``xsretrieval.models`` -- backbones, projection heads, and ensembling.

This package is the models / backbones layer of ``xsretrieval``. It exposes:

* the :class:`~xsretrieval.models.backbones.base.Backbone` interface, the string
  :data:`~xsretrieval.models.backbones.base.REGISTRY`,
  :func:`~xsretrieval.models.backbones.base.register_backbone`, and
  :func:`~xsretrieval.models.backbones.base.get_backbone` (with offline
  fallback);
* the concrete backbones (:class:`~xsretrieval.models.backbones.fallback.
  FallbackBackbone`, :class:`~xsretrieval.models.backbones.timm_backbone.
  TimmBackbone`, :class:`~xsretrieval.models.backbones.dinov2.DINOv2Backbone`,
  :class:`~xsretrieval.models.backbones.clip_backbones.OpenCLIPBackbone` /
  :class:`~xsretrieval.models.backbones.clip_backbones.RemoteCLIPBackbone`,
  :class:`~xsretrieval.models.backbones.dofa.DOFABackbone`,
  :class:`~xsretrieval.models.backbones.croma.CROMABackbone`);
* the :class:`~xsretrieval.models.ensemble.EnsembleBackbone` and
  :func:`~xsretrieval.models.ensemble.concat_and_renorm` helper;
* the trainable :class:`~xsretrieval.models.projection.ProjectionHeads`,
  :class:`~xsretrieval.models.projection.LoRALinear` adapter, and
  :func:`~xsretrieval.models.projection.freeze` /
  :func:`~xsretrieval.models.projection.unfreeze` utilities.

Bare-numpy safe: ``import xsretrieval.models`` works without torch. The
torch-only symbols (:class:`ProjectionHeads`, :class:`LoRALinear`) are
re-exported lazily (:pep:`562`); referencing them is what (lazily) requires
torch -- importing the package does not.
"""

from __future__ import annotations

from typing import Any

# Backbones + registry (numpy-safe; heavy deps stay lazy inside methods).
from .backbones import (
    REGISTRY,
    Backbone,
    CROMABackbone,
    DINOv2Backbone,
    DOFABackbone,
    FallbackBackbone,
    OpenCLIPBackbone,
    PrecomputedBackbone,
    RemoteCLIPBackbone,
    TimmBackbone,
    get_backbone,
    l2_normalize,
    register_backbone,
)

# Ensemble (numpy-safe).
from .ensemble import EnsembleBackbone, concat_and_renorm

# Torch-free projection utilities (the classes themselves are lazy; see below).
from .projection import freeze, unfreeze

__all__ = [
    # interface / registry
    "Backbone",
    "REGISTRY",
    "register_backbone",
    "get_backbone",
    "l2_normalize",
    # concrete backbones
    "FallbackBackbone",
    "PrecomputedBackbone",
    "TimmBackbone",
    "DINOv2Backbone",
    "OpenCLIPBackbone",
    "RemoteCLIPBackbone",
    "DOFABackbone",
    "CROMABackbone",
    # ensemble
    "EnsembleBackbone",
    "concat_and_renorm",
    # projection / adapters
    "ProjectionHeads",
    "LoRALinear",
    "freeze",
    "unfreeze",
]

# Names that live in ``projection`` and must stay lazy (torch-backed classes).
_LAZY_PROJECTION = {"ProjectionHeads", "LoRALinear"}


def __getattr__(name: str) -> Any:  # PEP 562: keep torch-only classes lazy
    """Resolve :class:`ProjectionHeads` / :class:`LoRALinear` on first access.

    Re-exports the lazily-built torch classes from
    :mod:`xsretrieval.models.projection` without importing torch at package
    import time.
    """
    if name in _LAZY_PROJECTION:
        from . import projection

        return getattr(projection, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - cosmetic
    return sorted(set(__all__) | set(globals()))
