"""Tests for the per-modality whitening / modality-gap remover.

``PerModalityWhitener`` (``xsretrieval.alignment.whitening``) implements the
highest-ROI modality-gap remedy from ``research/03`` Section 13: per-modality
mean-centering (+ PCA whitening) followed by L2-normalization, so optical / SAR /
MS embedding cones overlap and cross-modal cosine similarities become
comparable.

This test verifies the three properties the eval depends on, against the real
API discovered in the alignment module:

* ``fit(emb_by_mod: dict[Modality, ndarray])`` / ``transform(emb, modality)``.
* transformed outputs are L2-normalized (unit rows);
* on a constructed example with a deliberate inter-modal offset, the transform
  **shrinks the gap** between the two modalities' mean embeddings.

It skips gracefully if the alignment module is unavailable. It needs only numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from xsretrieval.data.modalities import Modality

whitening = pytest.importorskip(
    "xsretrieval.alignment.whitening",
    reason="alignment team's whitening module not available yet",
)
PerModalityWhitener = whitening.PerModalityWhitener


def _make_two_modality_embeddings(
    d: int = 16, n_per: int = 96, gap: float = 5.0, seed: int = 0
):
    """Two modality clouds sharing semantic structure but offset by a gap.

    Each modality is an isotropic Gaussian blob; the SAR cloud is shifted by a
    constant offset vector of norm ~``gap`` to simulate the modality gap. Both
    share the same underlying coordinate distribution so removing the offset
    should make them overlap.

    Returns
    -------
    (emb_by_mod, emb_opt, emb_sar) where ``emb_by_mod`` is the
    ``{Modality: ndarray}`` dict the whitener's ``fit`` expects.
    """
    rng = np.random.default_rng(seed)
    emb_opt = rng.normal(size=(n_per, d)).astype(np.float32)
    offset = np.zeros(d, dtype=np.float32)
    offset[0] = gap  # constant inter-modal offset along one axis
    emb_sar = (rng.normal(size=(n_per, d)) + offset).astype(np.float32)
    emb_by_mod = {Modality.OPTICAL_RGB: emb_opt, Modality.SAR: emb_sar}
    return emb_by_mod, emb_opt, emb_sar


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def _cone_centroid_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between the two modality cones' centroids on the hypersphere.

    Each modality's embeddings are L2-normalized to unit rows, then we take the
    **mean of those unit rows** (the cone centroid) and measure the Euclidean
    distance between the two centroids. This is the direct, robust probe for the
    modality gap (research §13): the gap is precisely the constant offset between
    the two cones' centroids. Per-modality mean-centering collapses each cone's
    centroid toward the origin, so this distance shrinks sharply.

    (Note: re-normalizing the centroid itself is *not* a valid probe -- once a
    centroid sits near the origin its direction is dominated by sampling noise.)
    """
    centroid_a = _l2(a).mean(axis=0)
    centroid_b = _l2(b).mean(axis=0)
    return float(np.linalg.norm(centroid_a - centroid_b))


def test_whitener_shapes_and_l2_norm() -> None:
    whitener = PerModalityWhitener()
    emb_by_mod, emb_opt, emb_sar = _make_two_modality_embeddings()

    whitener.fit(emb_by_mod)
    out_opt = np.asarray(whitener.transform(emb_opt, Modality.OPTICAL_RGB))
    out_sar = np.asarray(whitener.transform(emb_sar, Modality.SAR))

    # Row count preserved per modality.
    assert out_opt.shape[0] == emb_opt.shape[0]
    assert out_sar.shape[0] == emb_sar.shape[0]
    assert out_opt.ndim == 2 and out_sar.ndim == 2

    for out in (out_opt, out_sar):
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(
            norms, np.ones_like(norms), rtol=1e-4, atol=1e-4,
            err_msg="whitener output rows must be L2-normalized (unit norm)",
        )


def test_whitener_fit_transform_dict_l2_norm() -> None:
    """The convenience ``fit_transform`` returns a per-modality dict of units."""
    whitener = PerModalityWhitener()
    emb_by_mod, _, _ = _make_two_modality_embeddings()
    out = whitener.fit_transform(emb_by_mod)
    assert set(out.keys()) == set(emb_by_mod.keys())
    for arr in out.values():
        arr = np.asarray(arr)
        norms = np.linalg.norm(arr, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), rtol=1e-4, atol=1e-4)


def test_whitener_shrinks_cross_modal_mean_gap() -> None:
    whitener = PerModalityWhitener()
    emb_by_mod, emb_opt, emb_sar = _make_two_modality_embeddings(gap=5.0)

    before = _cone_centroid_gap(emb_opt, emb_sar)

    whitener.fit(emb_by_mod)
    out_opt = np.asarray(whitener.transform(emb_opt, Modality.OPTICAL_RGB))
    out_sar = np.asarray(whitener.transform(emb_sar, Modality.SAR))
    after = _cone_centroid_gap(out_opt, out_sar)

    assert after < before, (
        f"per-modality centering should shrink the cross-modal cone gap, "
        f"but gap went {before:.4f} -> {after:.4f}"
    )
    # The deliberate offset should be largely removed (at least halved).
    assert after < 0.5 * before, (
        f"expected the gap to at least halve; {before:.4f} -> {after:.4f}"
    )


def test_mean_center_per_modality_helper_shrinks_gap() -> None:
    """The cheap ``mean_center_per_modality`` helper also closes the gap."""
    fn = getattr(whitening, "mean_center_per_modality", None)
    if fn is None:
        pytest.skip("mean_center_per_modality not implemented")
    _, emb_opt, emb_sar = _make_two_modality_embeddings(gap=5.0)
    emb = np.concatenate([emb_opt, emb_sar], axis=0)
    mods = [Modality.OPTICAL_RGB] * len(emb_opt) + [Modality.SAR] * len(emb_sar)

    before = _cone_centroid_gap(emb_opt, emb_sar)
    out = np.asarray(fn(emb, mods))
    out_opt = out[: len(emb_opt)]
    out_sar = out[len(emb_opt):]
    after = _cone_centroid_gap(out_opt, out_sar)
    assert after < before
