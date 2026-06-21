"""Backbone subpackage: abstract interface, registry, and concrete backbones.

Everything here imports cleanly on a **bare-numpy** install -- the concrete
backbone classes only use numpy at import time and lazy-import their heavy
dependencies (``torch``/``transformers``/``timm``/``open_clip``/
``huggingface_hub``) inside methods. So ``import xsretrieval.models.backbones``
never pulls in torch.

Public API
----------
* :class:`Backbone` -- abstract base (embedding contract, channel adaptation).
* :data:`REGISTRY`, :func:`register_backbone`, :func:`get_backbone` -- string
  registry with offline fallback.
* :func:`l2_normalize` -- shared numpy L2-normalization helper.
* Concrete backbones: :class:`FallbackBackbone`, :class:`TimmBackbone`,
  :class:`DINOv2Backbone`, :class:`OpenCLIPBackbone`, :class:`RemoteCLIPBackbone`,
  :class:`DOFABackbone`, :class:`CROMABackbone`.
"""

from __future__ import annotations

from .base import (
    REGISTRY,
    Backbone,
    get_backbone,
    l2_normalize,
    register_backbone,
)
from .clip_backbones import OpenCLIPBackbone, RemoteCLIPBackbone
from .croma import CROMABackbone
from .dinov2 import DINOv2Backbone
from .dofa import DOFABackbone
from .fallback import FallbackBackbone
from .precomputed import PrecomputedBackbone
from .timm_backbone import TimmBackbone

__all__ = [
    # base / registry
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
]
