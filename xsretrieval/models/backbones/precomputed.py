"""Pre-encoded "embedding" backbone (``"precomputed"``).

:class:`PrecomputedBackbone` is a trivial pass-through backbone: it treats each
input "image" as an **already-encoded feature vector** and simply flattens and
L2-normalizes it. It exists so the rest of the pipeline (whitening → index →
evaluation) can be exercised on embeddings that were produced *elsewhere* — e.g.

* a real foundation model run offline and cached to ``(D, 1, 1)`` arrays, or
* the synthetic embedding generator
  :func:`~xsretrieval.data.synthetic.make_synthetic_embeddings`, which emits
  vectors carrying an explicit **modality gap**.

The embedding substrate is what lets the smoke test demonstrate the
per-modality-whitening win *honestly*: foundation encoders place each modality
in its own offset cone, and whitening removes that offset. The numpy
``FallbackBackbone`` (hand-crafted image statistics) has no such clean constant
gap, so whitening cannot help there; this backbone provides the realistic
regime without requiring a multi-gigabyte model download.

Input convention: any ``(C, H, W)`` array is flattened to ``C*H*W`` features
(so a ``(D, 1, 1)`` "image" round-trips to a ``D``-vector). Pure numpy; no heavy
dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import Backbone, register_backbone

if TYPE_CHECKING:  # pragma: no cover - typing only
    from xsretrieval.data.modalities import Modality

__all__ = ["PrecomputedBackbone"]


@register_backbone("precomputed", aliases=["embedding", "identity", "passthrough"])
class PrecomputedBackbone(Backbone):
    """Pass-through backbone that returns pre-encoded embeddings verbatim.

    Parameters
    ----------
    embed_dim:
        Expected feature width (informational; the actual width is whatever the
        flattened input provides). Default ``256``.
    device:
        Accepted for interface parity (computation is numpy/CPU).
    image_size:
        Accepted and ignored (configs may pass a shared ``image_size``).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        device: str = "cpu",
        image_size: int | None = None,
    ) -> None:
        self.name = "precomputed"
        self.embed_dim = int(embed_dim)
        self.device = device
        self.image_size = image_size
        self.supported_modalities = set()  # all modalities

    def _embed_batch(
        self, batch: np.ndarray, modality: "Modality | str | None"
    ) -> np.ndarray:
        """Flatten each ``(C, H, W)`` item to a feature vector (L2-norm by base)."""
        arr = np.asarray(batch, dtype=np.float32)
        feats = arr.reshape(arr.shape[0], -1)
        # Keep embed_dim in sync with the data actually seen (so a downstream
        # index built from these vectors gets the right dimensionality).
        self.embed_dim = int(feats.shape[1])
        return feats
