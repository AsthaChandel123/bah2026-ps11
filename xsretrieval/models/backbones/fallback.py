"""Offline, dependency-free fallback backbone (``"hashfeat"``).

:class:`FallbackBackbone` is the *always-works* feature extractor. It uses
**only numpy** -- no model downloads, no torch required -- and is fully
deterministic. Its job is twofold:

1. Be the safety net :func:`xsretrieval.models.backbones.base.get_backbone`
   falls back to when a real backbone's weights/dependencies are unavailable,
   so the retrieval pipeline never hard-crashes (CPU-only, offline).
2. Produce embeddings that are *meaningfully above chance* on the synthetic
   smoke test -- i.e. they must preserve scene/class structure, not just be
   random hashes. Two images of the same class should land closer than two of
   different classes.

Feature design
--------------
For an input image ``(C, H, W)`` we build a hand-crafted descriptor that
captures appearance statistics at multiple spatial scales, then compress it
with a fixed seeded Gaussian random projection (a Johnson-Lindenstrauss style
embedding that approximately preserves distances). Components, all per-channel:

* **Multi-scale average pooling** over ``1x1``, ``2x2`` and ``4x4`` grids of the
  per-cell **mean** and **std** -- a coarse spatial layout / texture-energy
  signature (1+4+16 = 21 cells x 2 stats per channel).
* **Gradient / edge energy**: mean absolute horizontal and vertical finite
  differences per channel -- structure / boundary content.
* **Channel-correlation summary**: the flattened upper triangle of the
  per-image channel correlation matrix -- inter-band relationships (e.g. how
  SAR VV/VH or MS bands co-vary), which is discriminative across land cover.

The raw descriptor length depends on ``C``; we therefore project it to a fixed
``embed_dim`` with a deterministic Gaussian matrix keyed by ``(raw_dim,
embed_dim, seed)`` and L2-normalize. Any channel count is accepted (RGB=3,
multispectral=13, SAR=2, or anything else), so ``supported_modalities`` is
"all".

An optional torch path exists only to accept ``torch.Tensor`` inputs; the math
is pure numpy regardless.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .base import Backbone, register_backbone

if TYPE_CHECKING:  # pragma: no cover - typing only
    from xsretrieval.data.modalities import Modality

logger = logging.getLogger(__name__)

__all__ = ["FallbackBackbone"]


@register_backbone("fallback", aliases=["hashfeat", "offline", "numpy"])
class FallbackBackbone(Backbone):
    """Deterministic, numpy-only feature extractor.

    Parameters
    ----------
    embed_dim:
        Output embedding width. Default ``256``.
    seed:
        Seed for the fixed random projection (and any internal randomness).
        Fixed across calls/instances so embeddings are reproducible and
        comparable. Default ``1234``.
    pool_grids:
        Spatial pooling grid sizes for the multi-scale statistics. Default
        ``(1, 2, 4)``.
    image_size:
        Unused for computation (descriptor is resolution-robust) but accepted so
        configs can pass a common ``image_size`` to every backbone.
    device:
        Accepted for interface parity; computation is always on CPU/numpy.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        seed: int = 1234,
        pool_grids: tuple[int, ...] = (1, 2, 4),
        image_size: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.name = "fallback"
        self.embed_dim = int(embed_dim)
        self.seed = int(seed)
        self.pool_grids = tuple(int(g) for g in pool_grids)
        self.image_size = image_size
        self.device = device
        # ``set()`` (empty) == "all modalities supported" per the base contract.
        self.supported_modalities = set()
        # Projection matrices are created lazily per raw-descriptor width and
        # cached, since the width depends on the channel count of the input.
        self._proj_cache: dict[int, np.ndarray] = {}

    # -- projection ---------------------------------------------------------
    def _projection(self, raw_dim: int) -> np.ndarray:
        """Return a cached ``(raw_dim, embed_dim)`` Gaussian projection.

        Deterministic in ``(raw_dim, embed_dim, seed)``. Columns are scaled by
        ``1/sqrt(embed_dim)`` (standard JL scaling) so the projection
        approximately preserves inner products.
        """
        proj = self._proj_cache.get(raw_dim)
        if proj is None:
            # Mix raw_dim into the seed so different widths get different (but
            # reproducible) matrices.
            rng = np.random.default_rng(self.seed + 1_000_003 * raw_dim)
            proj = rng.standard_normal((raw_dim, self.embed_dim)).astype(np.float32)
            proj /= np.sqrt(self.embed_dim, dtype=np.float32)
            self._proj_cache[raw_dim] = proj
        return proj

    # -- feature extraction -------------------------------------------------
    @staticmethod
    def _multiscale_pool(batch: np.ndarray, grids: tuple[int, ...]) -> np.ndarray:
        """Per-channel multi-scale mean+std pooling.

        Parameters
        ----------
        batch:
            ``(B, C, H, W)`` ``float32``.
        grids:
            Grid sizes; for grid ``g`` the image is split into ``g x g`` cells
            and the mean and std of each cell (per channel) are recorded.

        Returns
        -------
        numpy.ndarray
            ``(B, C * sum(g**2) * 2)`` features.
        """
        b, c, h, w = batch.shape
        feats: list[np.ndarray] = []
        for g in grids:
            # Cell boundaries (handle non-divisible sizes via linspace splits).
            ys = np.linspace(0, h, g + 1).astype(int)
            xs = np.linspace(0, w, g + 1).astype(int)
            for i in range(g):
                y0, y1 = ys[i], max(ys[i + 1], ys[i] + 1)
                for j in range(g):
                    x0, x1 = xs[j], max(xs[j + 1], xs[j] + 1)
                    cell = batch[:, :, y0:y1, x0:x1]  # (B, C, ch, cw)
                    flat = cell.reshape(b, c, -1)
                    feats.append(flat.mean(axis=2))  # (B, C)
                    feats.append(flat.std(axis=2))  # (B, C)
        # Concatenate along feature axis -> (B, C * n_cells * 2).
        return np.concatenate(feats, axis=1).astype(np.float32)

    @staticmethod
    def _gradient_energy(batch: np.ndarray) -> np.ndarray:
        """Per-channel mean |dx| and |dy| edge energy. -> ``(B, 2C)``."""
        b, c, h, w = batch.shape
        if w >= 2:
            dx = np.abs(np.diff(batch, axis=3)).reshape(b, c, -1).mean(axis=2)
        else:
            dx = np.zeros((b, c), dtype=np.float32)
        if h >= 2:
            dy = np.abs(np.diff(batch, axis=2)).reshape(b, c, -1).mean(axis=2)
        else:
            dy = np.zeros((b, c), dtype=np.float32)
        return np.concatenate([dx, dy], axis=1).astype(np.float32)

    @staticmethod
    def _channel_correlation(batch: np.ndarray, max_pairs: int = 64) -> np.ndarray:
        """Upper-triangle of the per-image channel correlation matrix.

        For ``C`` channels there are ``C*(C-1)/2`` unordered pairs. To keep the
        descriptor a fixed, bounded size we record up to ``max_pairs`` of them
        (the first pairs in row-major upper-triangular order); fewer channels
        are zero-padded to ``max_pairs``. ``C == 1`` yields all zeros.

        Returns
        -------
        numpy.ndarray
            ``(B, max_pairs)`` features.
        """
        b, c, h, w = batch.shape
        out = np.zeros((b, max_pairs), dtype=np.float32)
        if c < 2:
            return out
        flat = batch.reshape(b, c, -1)  # (B, C, HW)
        mean = flat.mean(axis=2, keepdims=True)
        std = flat.std(axis=2, keepdims=True) + 1e-6
        norm = (flat - mean) / std  # zero-mean, unit-std per channel
        n = norm.shape[2]
        # Correlation matrix per image: (B, C, C).
        corr = np.einsum("bik,bjk->bij", norm, norm) / float(n)
        # Gather upper-triangular pairs (excluding the diagonal).
        iu, ju = np.triu_indices(c, k=1)
        pairs = corr[:, iu, ju]  # (B, n_pairs)
        k = min(max_pairs, pairs.shape[1])
        out[:, :k] = pairs[:, :k]
        return out

    def _descriptor(self, batch: np.ndarray) -> np.ndarray:
        """Build the full hand-crafted descriptor for *batch*. -> ``(B, raw)``."""
        pooled = self._multiscale_pool(batch, self.pool_grids)
        grad = self._gradient_energy(batch)
        corr = self._channel_correlation(batch)
        desc = np.concatenate([pooled, grad, corr], axis=1).astype(np.float32)
        # Robust per-feature scaling: standardize within the batch is unsafe for
        # B==1, so instead squash with a stable log-modulus that tames large
        # dynamic ranges (e.g. SAR dB vs reflectance) while preserving sign.
        desc = np.sign(desc) * np.log1p(np.abs(desc))
        return desc

    # -- Backbone API -------------------------------------------------------
    def _embed_batch(
        self, batch: np.ndarray, modality: "Modality | str | None"
    ) -> np.ndarray:
        """Compute raw ``(B, embed_dim)`` features (L2-norm applied by base)."""
        batch = np.asarray(batch, dtype=np.float32)
        # Replace non-finite values defensively (NaN/inf from upstream).
        if not np.isfinite(batch).all():
            batch = np.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)
        desc = self._descriptor(batch)  # (B, raw_dim)
        proj = self._projection(desc.shape[1])  # (raw_dim, embed_dim)
        feats = desc @ proj  # (B, embed_dim)
        return feats.astype(np.float32)
