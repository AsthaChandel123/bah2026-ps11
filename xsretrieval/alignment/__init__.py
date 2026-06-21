"""Cross-modal alignment: training losses and modality-gap whitening.

Public API
----------
Losses (``xsretrieval.alignment.losses`` — torch lazy-imported at call time):

* :func:`symmetric_infonce`        — CLIP-style symmetric cross-modal InfoNCE.
* :func:`info_nce`                 — one-directional InfoNCE.
* :func:`SubCenterArcFace`         — sub-center angular-margin loss (factory).
* :func:`batch_hard_triplet`       — batch-hard (cross-modal) triplet loss.
* :func:`multi_similarity_loss`    — Multi-Similarity loss.
* :func:`smooth_ap`                — differentiable AP surrogate (rank-direct).
* :func:`CrossModalRetrievalLoss`  — combined ``1.5*InfoNCE + 1*ArcFace +
  0.5*Triplet`` recipe (factory).

Whitening (``xsretrieval.alignment.whitening`` — pure numpy):

* :class:`PerModalityWhitener`     — per-modality mean-center + PCA whitening
  (+ optional top-PC removal). The highest-ROI modality-gap fix.
* :class:`GlobalWhitener`          — single global whitening transform.
* :func:`mean_center_per_modality` — cheapest GR-CLIP mean-centering remedy.

The losses module imports torch lazily, so importing this package never requires
torch/faiss; the whitening utilities are numpy-only.
"""

from __future__ import annotations

from xsretrieval.alignment.losses import (
    CrossModalRetrievalLoss,
    SubCenterArcFace,
    batch_hard_triplet,
    info_nce,
    multi_similarity_loss,
    smooth_ap,
    symmetric_infonce,
)
from xsretrieval.alignment.whitening import (
    GlobalWhitener,
    PerModalityWhitener,
    mean_center_per_modality,
)
from xsretrieval.alignment.trainer import TrainResult, train_projection

__all__ = [
    # losses
    "symmetric_infonce",
    "info_nce",
    "SubCenterArcFace",
    "batch_hard_triplet",
    "multi_similarity_loss",
    "smooth_ap",
    "CrossModalRetrievalLoss",
    # whitening
    "PerModalityWhitener",
    "GlobalWhitener",
    "mean_center_per_modality",
    # training
    "train_projection",
    "TrainResult",
]
