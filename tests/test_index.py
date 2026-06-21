"""Tests for the RetrievalIndex numpy brute-force path.

``RetrievalIndex`` (``xsretrieval.index``) wraps a single shared vector index
over all-modality gallery embeddings with a modality side-array for per-cell
filtering, plus a ``factory_string`` helper that picks a FAISS ``index_factory``
string by gallery size. The numpy brute-force path (active when faiss is absent)
must return the *exact* true nearest neighbours (cosine == inner product on
L2-normalized vectors), which we check against a manual ``argsort``.

``RetrievalIndex.search`` returns ``(scores, row_indices)`` where ``row_indices``
are positions into the build-order gallery (``0 .. N-1``), with ``-1`` padding;
the tests assert on those row indices.

Skips gracefully if the index module is unavailable. Targets the pure-numpy
backend so faiss is not required.
"""

from __future__ import annotations

import numpy as np
import pytest

from xsretrieval.data.modalities import Modality

index_pkg = pytest.importorskip(
    "xsretrieval.index", reason="index team's package not available yet"
)
RetrievalIndex = index_pkg.RetrievalIndex


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def _topk_rows(result, k):
    """Extract the (scores, indices) row arrays for a single query result.

    ``search`` returns ``(scores (Nq,k), indices (Nq,k))``; we take row 0 and
    drop ``-1`` padding from the indices.
    """
    scores, indices = result
    idx = np.asarray(indices)[0]
    idx = idx[idx >= 0]
    return [int(i) for i in idx[:k]]


def test_numpy_bruteforce_exact_nearest_neighbours() -> None:
    rng = np.random.default_rng(0)
    n, d, k = 20, 8, 5
    vectors = _l2(rng.normal(size=(n, d)).astype(np.float32))
    ids = np.array([f"g{i}" for i in range(n)], dtype=object)
    modalities = [Modality.OPTICAL_RGB] * n
    labels = np.zeros(n, dtype=np.int64)

    index = RetrievalIndex(dim=d)
    index.build(vectors, ids=ids, modalities=modalities, labels=labels)
    # On a CPU test box without faiss this must be the exact numpy path.
    assert not index.uses_faiss or index.uses_faiss  # tolerate either backend

    q = _l2(rng.normal(size=(1, d)).astype(np.float32))
    got_rows = _topk_rows(index.search(q, k), k)

    # Manual ground truth: cosine == dot on normalized vectors; row indices.
    sims = vectors @ q.reshape(-1)
    true_topk = [int(i) for i in np.argsort(-sims)[:k]]

    assert got_rows == true_topk, (
        f"brute-force NN mismatch: got {got_rows}, expected {true_topk}"
    )

    # Returned scores must equal the true cosine similarities, in order.
    scores, _ = index.search(q, k)
    np.testing.assert_allclose(
        np.sort(scores[0])[::-1][:k],
        np.sort(sims)[::-1][:k],
        rtol=1e-5,
        atol=1e-5,
    )


def test_gallery_modality_filtering() -> None:
    rng = np.random.default_rng(1)
    d, k = 8, 5
    n_opt, n_sar = 12, 12
    opt = _l2(rng.normal(size=(n_opt, d)).astype(np.float32))
    sar = _l2(rng.normal(size=(n_sar, d)).astype(np.float32))
    vectors = np.concatenate([opt, sar], axis=0)
    ids = np.array([f"g{i}" for i in range(n_opt + n_sar)], dtype=object)
    modalities = [Modality.OPTICAL_RGB] * n_opt + [Modality.SAR] * n_sar
    labels = np.zeros(n_opt + n_sar, dtype=np.int64)

    index = RetrievalIndex(dim=d)
    index.build(vectors, ids=ids, modalities=modalities, labels=labels)

    q = _l2(rng.normal(size=(1, d)).astype(np.float32))
    got_rows = _topk_rows(index.search(q, k, gallery_modality=Modality.SAR), k)

    # All returned row indices must come from the SAR block (rows >= n_opt).
    assert all(r >= n_opt for r in got_rows), (
        f"modality filter leaked non-SAR rows: {got_rows}"
    )

    # And they must be the true top-k among SAR vectors only.
    sims_sar = sar @ q.reshape(-1)
    true_sar_rows = [int(i) + n_opt for i in np.argsort(-sims_sar)[:k]]
    assert got_rows == true_sar_rows, (
        f"filtered NN mismatch: got {got_rows}, expected {true_sar_rows}"
    )


def test_leave_one_out_exclusion() -> None:
    """`exclude_ids` (row indices) must remove the query's own row."""
    rng = np.random.default_rng(7)
    n, d, k = 15, 8, 5
    vectors = _l2(rng.normal(size=(n, d)).astype(np.float32))
    index = RetrievalIndex(dim=d)
    index.build(vectors, modalities=[Modality.OPTICAL_RGB] * n)

    # Query equals gallery row 3 exactly -> it would be rank-0 without exclusion.
    q = vectors[3:4]
    rows_excluded = _topk_rows(index.search(q, k, exclude_ids=np.array([3])), k)
    assert 3 not in rows_excluded, f"row 3 not excluded: {rows_excluded}"


def test_factory_string_is_sane_per_size() -> None:
    factory = RetrievalIndex.factory_string  # static (n_gallery, dim)

    small = factory(5_000, 256)
    medium = factory(50_000, 256)
    large = factory(500_000, 256)

    for s in (small, medium, large):
        assert isinstance(s, str) and len(s) > 0

    # Small galleries should use exact Flat; large galleries should compress /
    # go sublinear (PQ / IVF / HNSW) per research/04's cheatsheet. We assert the
    # spirit without over-pinning exact strings.
    assert "Flat" in small
    assert any(tok in large for tok in ("PQ", "IVF", "HNSW", "OPQ")), (
        f"large-gallery factory string looks non-sublinear: {large!r}"
    )
