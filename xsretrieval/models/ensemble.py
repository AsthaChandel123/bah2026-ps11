"""Multi-backbone ensemble for robust cross-verification embeddings.

:class:`EnsembleBackbone` runs several :class:`~xsretrieval.models.backbones.
base.Backbone` instances on the same input, optionally whitens each backbone's
output (a hook point for the per-modality mean-centering / whitening that closes
the modality gap, research §13), concatenates the per-backbone embeddings, and
L2-normalizes the result into a single robust descriptor. The combined
``embed_dim`` is the sum of the members' dims.

The motivation (the "multi-backbone cross-verification move"): different
backbones capture complementary signal -- a wavelength-conditioned multimodal
model (DOFA), a contrastive radar-optical model (CROMA), an RGB specialist
(RemoteCLIP) -- and concatenating their normalized embeddings yields a
descriptor that is more discriminative and more robust to any single model's
blind spots, at the cost of a wider vector.

Pure numpy at the boundary: like every backbone, :meth:`embed` returns a plain
``(B, sum_dim)`` ``float32`` L2-normalized array. No heavy imports here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from .backbones.base import Backbone, l2_normalize

if TYPE_CHECKING:  # pragma: no cover - typing only
    from xsretrieval.data.modalities import Modality

logger = logging.getLogger(__name__)

__all__ = ["EnsembleBackbone", "concat_and_renorm"]

#: A whitening hook: ``(emb, backbone_name, modality) -> emb`` applied to each
#: backbone's (already L2-normalized) output before concatenation.
WhitenHook = Callable[[np.ndarray, str, Any], np.ndarray]


def concat_and_renorm(
    embeddings: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Concatenate per-backbone embeddings along the feature axis and renorm.

    Parameters
    ----------
    embeddings:
        Sequence of ``(B, d_k)`` ``float32`` arrays, all with the same ``B``.
    weights:
        Optional per-embedding scalar weights (length == ``len(embeddings)``)
        applied before concatenation, so a backbone can be up/down-weighted in
        the fused vector. Defaults to equal weight.

    Returns
    -------
    numpy.ndarray
        ``(B, sum_k d_k)`` ``float32`` array, L2-normalized along axis 1.
    """
    if not embeddings:
        raise ValueError("concat_and_renorm requires at least one embedding")
    arrs = [np.asarray(e, dtype=np.float32) for e in embeddings]
    batch = arrs[0].shape[0]
    for e in arrs:
        if e.ndim != 2:
            raise ValueError(f"each embedding must be 2-D, got shape {e.shape!r}")
        if e.shape[0] != batch:
            raise ValueError(
                "all embeddings must share batch size; got "
                f"{[a.shape for a in arrs]!r}"
            )
    if weights is not None:
        if len(weights) != len(arrs):
            raise ValueError("weights length must match number of embeddings")
        arrs = [a * float(w) for a, w in zip(arrs, weights)]
    concat = np.concatenate(arrs, axis=1)
    return l2_normalize(concat, axis=1)


class EnsembleBackbone(Backbone):
    """Concatenation ensemble over multiple backbones.

    Parameters
    ----------
    backbones:
        The member backbones. Each is called via its public :meth:`Backbone.
        embed`, so each contributes an L2-normalized block to the concatenation.
    weights:
        Optional per-backbone scalar weights applied before concatenation.
    whiten_hook:
        Optional callable ``(emb, backbone_name, modality) -> emb`` applied to
        each member's output before concatenation -- the place to plug in
        per-modality mean-centering / PCA-whitening (research §13). ``None``
        disables it.
    name:
        Optional name; defaults to ``"ensemble[<members>]"``.
    device:
        Advisory device string (members keep their own devices).
    """

    def __init__(
        self,
        backbones: Sequence[Backbone],
        weights: Sequence[float] | None = None,
        whiten_hook: WhitenHook | None = None,
        name: str | None = None,
        device: str = "cpu",
    ) -> None:
        if not backbones:
            raise ValueError("EnsembleBackbone requires at least one backbone")
        self.backbones = list(backbones)
        self.weights = list(weights) if weights is not None else None
        if self.weights is not None and len(self.weights) != len(self.backbones):
            raise ValueError("weights length must match number of backbones")
        self.whiten_hook = whiten_hook
        self.device = device
        self.embed_dim = int(sum(int(b.embed_dim) for b in self.backbones))
        members = ",".join(b.name for b in self.backbones)
        self.name = name or f"ensemble[{members}]"
        # Supported modalities = intersection if any member restricts; an empty
        # set on a member means "all", so it does not constrain the ensemble.
        restricting = [
            set(b.supported_modalities)
            for b in self.backbones
            if b.supported_modalities
        ]
        if restricting:
            inter = set.intersection(*restricting)
            self.supported_modalities = inter
        else:
            self.supported_modalities = set()

    # -- public override ----------------------------------------------------
    def embed(
        self, images: Any, modality: "Modality | str | None" = None
    ) -> np.ndarray:
        """Embed with every member, (optionally) whiten, concat, L2-normalize.

        Overrides :meth:`Backbone.embed` directly because the per-member
        normalization + concatenation *is* the ensemble's semantics (the base
        class's single-pass ``_embed_batch`` contract does not fit a multi-model
        concat). Returns ``(B, sum_dim)`` ``float32`` L2-normalized.
        """
        blocks: list[np.ndarray] = []
        for b in self.backbones:
            emb = b.embed(images, modality)  # (B, d_k), L2-normalized
            if self.whiten_hook is not None:
                emb = np.asarray(
                    self.whiten_hook(emb, b.name, modality), dtype=np.float32
                )
            blocks.append(emb)
        return concat_and_renorm(blocks, weights=self.weights)

    # -- abstract satisfaction ---------------------------------------------
    def _embed_batch(
        self, batch: np.ndarray, modality: "Modality | str | None"
    ) -> np.ndarray:
        # Not used (``embed`` is overridden) but required by the ABC. Provide a
        # consistent implementation in case a caller invokes it directly.
        blocks = [b._embed_batch(batch, modality) for b in self.backbones]
        blocks = [l2_normalize(np.asarray(x, np.float32), axis=1) for x in blocks]
        if self.weights is not None:
            blocks = [x * float(w) for x, w in zip(blocks, self.weights)]
        return np.concatenate(blocks, axis=1)
