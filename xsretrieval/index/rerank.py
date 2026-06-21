"""Re-ranking and query-expansion utilities (pure numpy).

After a fast/lossy first retrieval stage returns a candidate pool, these cheap
refinements *buy F1 back* by re-ordering only the small top-K (research note
``04_fast_retrieval.md`` §13).  Latency added is ``O(K * ...)`` with ``K`` small
(50-200), so per-query time barely moves while F1@5/@10 — especially for the
harder **cross-modal** queries — improves measurably.

Functions
---------
* :func:`exact_refine`          — exact cosine re-sort of a candidate pool
  (the cheapest, always-safe win after any approximate first stage).
* :func:`k_reciprocal_rerank`   — Zhong et al. CVPR'17 k-reciprocal encoding +
  Jaccard distance (the strongest training-free F1 booster).
* :func:`average_query_expansion` / :func:`alpha_qe` — query expansion by
  averaging the query with its top retrieved neighbours.

All operate on **L2-normalized** embeddings so that ``cosine == inner product``.
Everything is numpy; there is no torch / faiss dependency here.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "exact_refine",
    "k_reciprocal_rerank",
    "average_query_expansion",
    "alpha_qe",
]


def _as_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return ``(x_2d, was_1d)`` coercing a 1-D query into a single row."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x[None, :], True
    return x, False


def exact_refine(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    candidate_indices: np.ndarray,
    k: int,
) -> np.ndarray:
    """Re-sort a candidate pool by **exact** cosine similarity, keep top-``k``.

    The canonical recovery step after a compressed/approximate first stage
    (PQ / IVF / HNSW / binary hashing): recompute the true inner product on just
    the candidate pool and re-order.  Restores ``IndexFlat``-level F1 at trivial
    cost — ``O(|pool| * d)`` per query (research §13, *exact-distance refine*).

    Parameters
    ----------
    query_emb:
        ``(d,)`` or ``(1, d)`` L2-normalized query embedding.
    gallery_emb:
        ``(N, d)`` L2-normalized gallery embeddings.
    candidate_indices:
        1-D array of gallery row indices forming the candidate pool (negative
        entries, e.g. faiss ``-1`` padding, are ignored).
    k:
        Number of refined neighbours to return.

    Returns
    -------
    np.ndarray
        1-D int64 array of up to ``k`` gallery indices, best-first.
    """
    q, _ = _as_2d(query_emb)
    cand = np.asarray(candidate_indices, dtype=np.int64).ravel()
    cand = cand[cand >= 0]
    if cand.size == 0:
        return np.empty(0, dtype=np.int64)
    sims = gallery_emb[cand] @ q[0]                      # (|pool|,)
    order = np.argsort(-sims, kind="stable")
    return cand[order[:k]].astype(np.int64)


def _jaccard_distance_block(
    feats: np.ndarray, query_feat: np.ndarray
) -> np.ndarray:
    """Jaccard distance between one k-reciprocal feature and a block of them.

    ``d_J(i, j) = 1 - sum_k min(V_i[k], V_j[k]) / sum_k max(V_i[k], V_j[k])``.
    """
    mins = np.minimum(feats, query_feat[None, :]).sum(axis=1)
    maxs = np.maximum(feats, query_feat[None, :]).sum(axis=1)
    maxs = np.maximum(maxs, 1e-12)
    return 1.0 - mins / maxs


def k_reciprocal_rerank(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    init_indices: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_: float = 0.3,
) -> np.ndarray:
    """k-reciprocal re-ranking (Zhong et al., CVPR 2017), numpy.

    Builds k-reciprocal neighbour sets, encodes them as sparse similarity
    features, expands by ``k2``-neighbour pooling, and combines the **Jaccard**
    distance of those features with the original cosine distance:

    .. math::

        d^*(q, g_i) = (1 - \\lambda)\\, d_{\\text{Jaccard}}(q, g_i)
                    + \\lambda\\, d_{\\text{orig}}(q, g_i)

    This is the strongest *training-free* F1 booster in the toolbox (Market-1501
    mAP 46.0 → 59.87 in the original paper).  To keep it cheap it is run **only
    on the candidate pool** ``init_indices`` (the naive full-gallery version is
    ``O(N^2)``), as recommended in research §13/§16.

    Parameters
    ----------
    query_emb:
        ``(d,)`` or ``(1, d)`` L2-normalized query embedding.
    gallery_emb:
        ``(N, d)`` L2-normalized gallery embeddings.
    init_indices:
        1-D array of candidate gallery indices from the first stage (best-first
        order is *not* required).  Negative entries are dropped.
    k1:
        Size of the k-reciprocal neighbourhood (paper default 20).
    k2:
        Local query-expansion neighbourhood size (paper default 6).
    lambda_:
        Trade-off between Jaccard and original distance (paper default 0.3).

    Returns
    -------
    np.ndarray
        1-D int64 array of the candidate indices re-ordered best-first.

    Notes
    -----
    We assemble a small ``(M+1) x (M+1)`` problem over ``{query} ∪ pool`` of size
    ``M = |pool|``, so cost is ``O(M^2 d + M^2 k1)`` — negligible for the
    ``M <= a few hundred`` pools used in practice.
    """
    q, _ = _as_2d(query_emb)
    cand = np.asarray(init_indices, dtype=np.int64).ravel()
    cand = cand[cand >= 0]
    # Deduplicate while preserving order (a candidate pool should be unique, but
    # be defensive).
    _, uniq_pos = np.unique(cand, return_index=True)
    cand = cand[np.sort(uniq_pos)]
    m = cand.size
    if m == 0:
        return np.empty(0, dtype=np.int64)
    if m == 1:
        return cand.astype(np.int64)

    # Effective k1 cannot exceed the pool size (index 0 is the query itself).
    k1_eff = int(min(k1, m))
    k2_eff = int(max(1, min(k2, m + 1)))

    # Stack query (row 0) with the candidate gallery vectors (rows 1..M).
    feats = np.vstack([q, gallery_emb[cand]]).astype(np.float32)  # (M+1, d)
    n = feats.shape[0]

    # Original cosine distance (query row only is needed for the final blend).
    sim = feats @ feats.T                                # (M+1, M+1)
    sim = np.clip(sim, -1.0, 1.0)
    orig_dist = 1.0 - sim                                # in [0, 2]

    # Rank lists by ascending distance (each row's nearest first).
    rank = np.argsort(orig_dist, axis=1, kind="stable")  # (M+1, M+1)

    # ----- k-reciprocal neighbour sets -----
    # initial_rank[i, :k1+1] are the (k1+1) nearest (including self at pos 0).
    def k_reciprocal_neigh(i: int, k1v: int) -> np.ndarray:
        forward = rank[i, : k1v + 1]
        # Backward check: keep j whose own (k1+1)-NN list contains i.
        recip = [int(j) for j in forward if i in rank[j, : k1v + 1]]
        return np.asarray(recip, dtype=np.int64)

    # Build the expanded k-reciprocal set per row, then the V (feature) matrix.
    V = np.zeros((n, n), dtype=np.float32)
    krnn_cache: list[np.ndarray] = []
    for i in range(n):
        krnn = k_reciprocal_neigh(i, k1_eff)
        krnn_cache.append(krnn)
    for i in range(n):
        krnn = krnn_cache[i].copy()
        # Expand: add half-k1 reciprocal neighbours of each member if they
        # overlap enough with the current set (Zhong et al. expansion rule).
        expansion = set(int(x) for x in krnn)
        half = int(round(k1_eff / 2.0)) + 1
        for j in krnn.tolist():
            cand_j = k_reciprocal_neigh(int(j), half)
            inter = np.intersect1d(cand_j, krnn, assume_unique=False)
            if cand_j.size != 0 and inter.size > (2.0 / 3.0) * cand_j.size:
                expansion.update(int(x) for x in cand_j.tolist())
        members = np.asarray(sorted(expansion), dtype=np.int64)
        # Gaussian-weight the members by their distance to i, then L1-normalise.
        w = np.exp(-orig_dist[i, members])
        w = w / max(w.sum(), 1e-12)
        V[i, members] = w

    # ----- local query expansion over k2 neighbours -----
    if k2_eff > 1:
        V_qe = np.zeros_like(V)
        for i in range(n):
            neigh = rank[i, :k2_eff]
            V_qe[i] = V[neigh].mean(axis=0)
        V = V_qe

    # ----- Jaccard distance from the query (row 0) to every pool member -----
    jacc = _jaccard_distance_block(V[1:], V[0])          # (M,)

    # ----- final blended distance and ordering -----
    final = (1.0 - lambda_) * jacc + lambda_ * orig_dist[0, 1:]
    order = np.argsort(final, kind="stable")
    return cand[order].astype(np.int64)


def average_query_expansion(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    indices: np.ndarray,
    top: int = 5,
) -> np.ndarray:
    """Average Query Expansion (AQE): mean of query + its top-``top`` neighbours.

    Returns a **new query embedding** (L2-normalized) that is the average of the
    original query and its ``top`` retrieved gallery descriptors; re-searching
    with it typically lifts recall on instance retrieval (research §13, *AQE*).

    Parameters
    ----------
    query_emb:
        ``(d,)`` or ``(1, d)`` L2-normalized query.
    gallery_emb:
        ``(N, d)`` L2-normalized gallery embeddings.
    indices:
        1-D array of retrieved gallery indices (best-first).
    top:
        Number of leading neighbours to average in.

    Returns
    -------
    np.ndarray
        ``(d,)`` L2-normalized expanded query embedding.
    """
    return alpha_qe(query_emb, gallery_emb, indices, top=top, alpha=0.0)


def alpha_qe(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    indices: np.ndarray,
    top: int = 5,
    alpha: float = 3.0,
) -> np.ndarray:
    r"""Alpha Query Expansion (α-QE): similarity-weighted query expansion.

    Generalises AQE by weighting each retrieved neighbour ``g_i`` by
    ``(q \cdot g_i)^\alpha`` before averaging (Radenović et al.).  ``alpha=0``
    recovers plain AQE (uniform weights).  α≈1–3 is the de-facto standard.

    Parameters
    ----------
    query_emb:
        ``(d,)`` or ``(1, d)`` L2-normalized query.
    gallery_emb:
        ``(N, d)`` L2-normalized gallery embeddings.
    indices:
        1-D array of retrieved gallery indices (best-first).
    top:
        Number of leading neighbours to use.
    alpha:
        Similarity weighting exponent (default 3.0).

    Returns
    -------
    np.ndarray
        ``(d,)`` L2-normalized expanded query embedding.
    """
    q, _ = _as_2d(query_emb)
    q0 = q[0]
    idx = np.asarray(indices, dtype=np.int64).ravel()
    idx = idx[idx >= 0][:top]
    if idx.size == 0:
        return q0.astype(np.float32)
    neighbours = gallery_emb[idx]                        # (t, d)
    sims = np.clip(neighbours @ q0, 0.0, 1.0)            # (t,)
    weights = sims ** float(alpha)                       # alpha=0 -> all ones
    pooled = (weights[:, None] * neighbours).sum(axis=0)
    expanded = q0 + pooled                               # include the query
    norm = np.linalg.norm(expanded)
    if norm < 1e-12:
        return q0.astype(np.float32)
    return (expanded / norm).astype(np.float32)
