"""Pure-numpy retrieval metrics for cross-modal satellite image retrieval.

This module is the **scored core** of ``xsretrieval`` for BAH 2026 PS-11: the
challenge is graded on **F1@5 / F1@10** (same-modal and cross-modal) plus
average retrieval time, so the correctness of these functions *is* the score.
Everything here is therefore implemented from first principles, with the exact
closed-form formula documented next to each metric and cross-checked in
``tests/test_metrics.py``.

Design constraints
------------------
* **Pure numpy.** No torch / faiss / pandas. Importing this module must never
  pull in a heavy dependency, so the metrics can be unit-tested with only
  ``numpy`` installed (the rest of the package may not be importable yet).
* **Plain-array boundary.** Inputs are numpy arrays of *labels* (or boolean
  relevance masks) plus plain Python scalars. The functions never see
  embeddings or a search index -- ranking has already happened upstream; here
  we only score a ranked list against its ground-truth relevant set.
* **Two relevance conventions** are supported everywhere (see below).

Relevance model
---------------
For a single query *q* and a ranked list of retrieved gallery items, an item is
**relevant** iff it belongs to the relevant set ``rel(q)``. PS-11 (`idea.md`)
defines relevance by "semantic class, geographic correspondence, or predefined
relevance labels", so this module supports both operational forms:

* **(a) class equality** -- pass ``retrieved_labels`` (the label of each
  retrieved item, in rank order) and the scalar ``query_label``; an item is
  relevant iff ``retrieved_labels[i] == query_label``.
* **(b) explicit relevance mask** -- pass a boolean ``relevant_mask`` aligned to
  the ranked list (``relevant_mask[i] is True`` iff the *i*-th retrieved item is
  relevant). This covers multi-label / geographic-correspondence relevance where
  "relevant" is not a single class equality.

In both cases ``R_q`` (``total_relevant``) is the **total** number of relevant
items for the query *in the gallery* -- not just those retrieved. It is supplied
by the caller because it depends on the gallery, which these functions never
see.

The F1@K recall-denominator caveat (critical -- it bounds the score)
-------------------------------------------------------------------
``F1@K`` is dominated by ``R_q`` relative to ``K``, not only by ranking quality
(see ``research/06_sota_evaluation.md`` Part 2B):

* If ``R_q >> K`` (a large class), even a *perfect* top-K gives recall
  ``K / R_q`` -> F1 is capped low (e.g. ``R_q=100, K=5`` => max ``F1@5 = 0.095``).
* If ``R_q < K`` (a small class) you cannot fill K slots with relevants, so
  precision drops (e.g. ``R_q=2, K=5`` perfect => ``F1@5 = 0.571``).

Two recall conventions are therefore offered via ``mode``:

* ``mode="raw"`` -- ``R@K = r_K / R_q``. The literal reading and what an
  automated grader most likely implements. **Use this as the reported number.**
* ``mode="capped"`` -- ``R@K = r_K / min(K, R_q)``. Removes the large-class cap
  so a perfect top-K -> recall 1; the *fair* number for internal model
  selection.

Every per-query metric below documents its exact formula and honours this
``mode`` switch where recall is involved.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

__all__ = [
    "RECALL_MODES",
    "relevance_from_labels",
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "average_precision",
    "ndcg_at_k",
    "reciprocal_rank",
    "batch_f1_at_k",
    "mean_metrics",
]

#: Supported recall-denominator conventions (see module docstring).
RECALL_MODES: tuple[str, ...] = ("raw", "capped")

# A "relevance specification" may be given either as a boolean/0-1 mask aligned
# to the ranked list, or implicitly via class labels + a query label.
ArrayLike = Union[np.ndarray, "list[float]", "list[int]", "list[bool]"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _as_relevance_mask(
    retrieved_labels: Optional[ArrayLike],
    query_label: Optional[int],
    relevant_mask: Optional[ArrayLike],
) -> np.ndarray:
    """Resolve the two relevance interfaces into a 1-D boolean ranked mask.

    Exactly one of the two forms must be supplied:

    * ``relevant_mask`` -- a boolean/0-1 array, already aligned to the ranked
      retrieval list (the generic relevance-set interface).
    * ``retrieved_labels`` + ``query_label`` -- class-equality relevance.

    Returns a contiguous 1-D ``bool`` array in rank order (rank 0 first).
    """
    if relevant_mask is not None:
        if retrieved_labels is not None or query_label is not None:
            raise ValueError(
                "Provide either relevant_mask OR (retrieved_labels + "
                "query_label), not both."
            )
        mask = np.asarray(relevant_mask)
        if mask.ndim != 1:
            raise ValueError(
                f"relevant_mask must be 1-D (rank order); got shape {mask.shape}"
            )
        return mask.astype(bool)

    if retrieved_labels is None or query_label is None:
        raise ValueError(
            "Supply relevance as either relevant_mask, or both "
            "retrieved_labels and query_label."
        )
    labels = np.asarray(retrieved_labels)
    if labels.ndim != 1:
        raise ValueError(
            f"retrieved_labels must be 1-D (rank order); got shape {labels.shape}"
        )
    return labels == query_label


def _validate_k(k: int) -> int:
    """Validate and normalise the cutoff ``k`` (must be a positive integer)."""
    k_int = int(k)
    if k_int <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return k_int


def _validate_mode(mode: str) -> str:
    if mode not in RECALL_MODES:
        raise ValueError(
            f"mode must be one of {RECALL_MODES!r}, got {mode!r}"
        )
    return mode


def _relevant_in_top_k(mask: np.ndarray, k: int) -> int:
    """``r_K`` -- number of relevant items among the first ``k`` ranks.

    If fewer than ``k`` items were retrieved, only the available ranks are
    counted (the missing slots contribute zero relevants, which is the correct
    behaviour for a short retrieval list).
    """
    if mask.size == 0:
        return 0
    return int(np.count_nonzero(mask[:k]))


# ---------------------------------------------------------------------------
# Public relevance helper
# ---------------------------------------------------------------------------
def relevance_from_labels(
    retrieved_labels: ArrayLike, query_label: int
) -> np.ndarray:
    """Build a boolean relevance mask from class equality (form (a)).

    Parameters
    ----------
    retrieved_labels:
        1-D array of the labels of the retrieved gallery items, **in rank
        order** (most similar first).
    query_label:
        The query's class label. An item is relevant iff its label equals this.

    Returns
    -------
    mask:
        1-D ``bool`` array, ``mask[i] == (retrieved_labels[i] == query_label)``.
    """
    return _as_relevance_mask(retrieved_labels, query_label, None)


# ---------------------------------------------------------------------------
# Precision@K
# ---------------------------------------------------------------------------
def precision_at_k(
    retrieved_labels: Optional[ArrayLike] = None,
    query_label: Optional[int] = None,
    k: int = 10,
    *,
    relevant_mask: Optional[ArrayLike] = None,
) -> float:
    r"""Precision@K -- fraction of the top-K that are relevant.

    Formula
    -------
    .. math:: P@K = \frac{r_K}{K}

    where ``r_K`` = number of relevant items among the top-K. Note the
    denominator is the fixed cutoff ``K`` (not the number retrieved): if the
    system returns fewer than ``K`` items the empty slots count as
    non-relevant, which correctly penalises short result lists.

    Relevance may be specified either by class equality
    (``retrieved_labels`` + ``query_label``) or by an explicit
    ``relevant_mask`` aligned to the ranked list.

    Parameters
    ----------
    retrieved_labels:
        1-D labels of retrieved items in rank order (form (a)).
    query_label:
        Query class label (form (a)).
    k:
        Positive cutoff.
    relevant_mask:
        Boolean ranked relevance mask (form (b)); mutually exclusive with the
        label form.

    Returns
    -------
    float in ``[0, 1]``.
    """
    k = _validate_k(k)
    mask = _as_relevance_mask(retrieved_labels, query_label, relevant_mask)
    r_k = _relevant_in_top_k(mask, k)
    return r_k / k


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------
def recall_at_k(
    retrieved_labels: Optional[ArrayLike] = None,
    query_label: Optional[int] = None,
    total_relevant: int = 0,
    k: int = 10,
    mode: str = "raw",
    *,
    relevant_mask: Optional[ArrayLike] = None,
) -> float:
    r"""Recall@K -- fraction of all relevant items captured in the top-K.

    Formulas (selected by ``mode``)
    -------------------------------
    * ``mode="raw"``    : :math:`R@K = r_K / R_q`
    * ``mode="capped"`` : :math:`R@K = r_K / \min(K, R_q)`

    where ``r_K`` = relevant in top-K and ``R_q`` = ``total_relevant`` (the
    total number of relevant items for the query in the gallery).

    The ``raw`` form is the literal definition (and what graders usually use);
    the ``capped`` form removes the structural "large-class cap" so a perfect
    top-K yields recall 1 regardless of class size (see module docstring's
    caveat). With ``R_q <= K`` the two coincide.

    Edge case: if ``total_relevant == 0`` the query has no relevant items;
    recall is undefined and we return ``0.0`` by convention.

    Returns
    -------
    float in ``[0, 1]``.
    """
    k = _validate_k(k)
    mode = _validate_mode(mode)
    total_relevant = int(total_relevant)
    if total_relevant < 0:
        raise ValueError(f"total_relevant must be >= 0, got {total_relevant}")
    if total_relevant == 0:
        return 0.0
    mask = _as_relevance_mask(retrieved_labels, query_label, relevant_mask)
    r_k = _relevant_in_top_k(mask, k)
    denom = total_relevant if mode == "raw" else min(k, total_relevant)
    return r_k / denom


# ---------------------------------------------------------------------------
# F1@K
# ---------------------------------------------------------------------------
def f1_at_k(
    retrieved_labels: Optional[ArrayLike] = None,
    query_label: Optional[int] = None,
    total_relevant: int = 0,
    k: int = 10,
    mode: str = "raw",
    *,
    relevant_mask: Optional[ArrayLike] = None,
) -> float:
    r"""F1@K -- harmonic mean of Precision@K and Recall@K.

    Definition (harmonic mean)
    --------------------------
    .. math:: F1@K = \frac{2 \cdot P@K \cdot R@K}{P@K + R@K}

    **Closed form for ``mode="raw"``** (the headline PS-11 metric):

    .. math:: F1@K = \frac{2 r_K}{K + R_q}

    which follows by substituting :math:`P@K = r_K/K` and
    :math:`R@K = r_K/R_q` into the harmonic mean. This module computes F1 via
    the harmonic mean of :func:`precision_at_k` and :func:`recall_at_k` (so the
    same ``mode`` governs the recall term), and ``tests/test_metrics.py``
    asserts it equals the closed form to floating-point tolerance.

    For ``mode="capped"`` the recall denominator is ``min(K, R_q)`` so the
    closed form generalises to :math:`F1@K = 2 r_K / (K + \min(K, R_q))`.

    Edge cases:

    * ``total_relevant == 0`` -> F1 is ``0.0`` (no relevant items to find).
    * ``P@K == 0`` and ``R@K == 0`` (no relevant retrieved) -> ``0.0`` (the
      harmonic-mean denominator would be zero; F1 is conventionally 0 there).

    Returns
    -------
    float in ``[0, 1]``.
    """
    k = _validate_k(k)
    mode = _validate_mode(mode)
    p = precision_at_k(
        retrieved_labels, query_label, k, relevant_mask=relevant_mask
    )
    r = recall_at_k(
        retrieved_labels,
        query_label,
        total_relevant,
        k,
        mode,
        relevant_mask=relevant_mask,
    )
    denom = p + r
    if denom == 0.0:
        return 0.0
    return 2.0 * p * r / denom


# ---------------------------------------------------------------------------
# Average Precision (AP)
# ---------------------------------------------------------------------------
def average_precision(
    retrieved_labels: Optional[ArrayLike] = None,
    query_label: Optional[int] = None,
    total_relevant: int = 0,
    k: Optional[int] = None,
    *,
    relevant_mask: Optional[ArrayLike] = None,
) -> float:
    r"""Average Precision for one query (the per-query term of mAP).

    Formula
    -------
    .. math::

        AP = \frac{1}{R_q} \sum_{i=1}^{n} \mathbb{1}[\text{rel}_i] \cdot P@i

    i.e. the mean of the precision values *evaluated at the ranks where a
    relevant item occurs*, normalised by the total number of relevant items
    ``R_q`` (``total_relevant``). Using ``R_q`` (rather than the number of
    relevants retrieved) is the standard TREC/`AP` convention: relevant items
    that never appear in the list contribute 0 to the sum, so a system that
    fails to retrieve some relevants is penalised.

    ``k`` optionally truncates the list to the top-``k`` ranks (AP@k). By
    default the whole ranked list is used.

    Edge case: ``total_relevant == 0`` -> ``0.0``.

    Returns
    -------
    float in ``[0, 1]``.
    """
    total_relevant = int(total_relevant)
    if total_relevant < 0:
        raise ValueError(f"total_relevant must be >= 0, got {total_relevant}")
    if total_relevant == 0:
        return 0.0
    mask = _as_relevance_mask(retrieved_labels, query_label, relevant_mask)
    if k is not None:
        mask = mask[: _validate_k(k)]
    if mask.size == 0:
        return 0.0

    rel = mask.astype(np.float64)
    # Cumulative number of relevants up to and including each rank.
    cum_rel = np.cumsum(rel)
    ranks = np.arange(1, rel.size + 1, dtype=np.float64)
    precision_at_each = cum_rel / ranks  # P@i for every rank i
    # Sum precision only at ranks that are relevant, then normalise by R_q.
    ap = float(np.sum(precision_at_each * rel) / total_relevant)
    return ap


# ---------------------------------------------------------------------------
# nDCG@K
# ---------------------------------------------------------------------------
def ndcg_at_k(
    retrieved_labels: Optional[ArrayLike] = None,
    query_label: Optional[int] = None,
    total_relevant: int = 0,
    k: int = 10,
    *,
    relevant_mask: Optional[ArrayLike] = None,
    gains: Optional[ArrayLike] = None,
) -> float:
    r"""Normalised Discounted Cumulative Gain at K.

    Formula (binary relevance, standard log2 discount)
    --------------------------------------------------
    .. math::

        DCG@K = \sum_{i=1}^{K} \frac{g_i}{\log_2(i + 1)}, \qquad
        nDCG@K = \frac{DCG@K}{IDCG@K}

    where ``g_i`` is the gain of the item at rank ``i`` (1 for a relevant item,
    0 otherwise in the binary case) and ``IDCG@K`` is the DCG of the ideal
    ranking -- all relevant items placed first. With binary relevance the ideal
    ranking has ``min(K, R_q)`` ones at the top, so

    .. math:: IDCG@K = \sum_{i=1}^{\min(K, R_q)} \frac{1}{\log_2(i+1)}.

    Graded relevance is supported via the optional ``gains`` argument (a
    per-rank gain vector aligned to the retrieval list); when given it overrides
    the binary relevance mask and the ideal DCG is computed from the
    sorted-descending gains.

    Edge cases: empty list or ``IDCG == 0`` (no relevant items) -> ``0.0``.

    Returns
    -------
    float in ``[0, 1]``.
    """
    k = _validate_k(k)

    if gains is not None:
        gain_vec = np.asarray(gains, dtype=np.float64)
        if gain_vec.ndim != 1:
            raise ValueError(
                f"gains must be 1-D (rank order); got shape {gain_vec.shape}"
            )
        ideal_pool = gain_vec.copy()
    else:
        mask = _as_relevance_mask(retrieved_labels, query_label, relevant_mask)
        gain_vec = mask.astype(np.float64)
        # Build the ideal-gain pool of length R_q (all relevant items = gain 1)
        # so that IDCG reflects relevants that exist in the gallery but were not
        # retrieved within the list.
        total_relevant = int(total_relevant)
        n_ideal = total_relevant if total_relevant > 0 else int(gain_vec.sum())
        ideal_pool = np.ones(n_ideal, dtype=np.float64)

    if gain_vec.size == 0 and ideal_pool.size == 0:
        return 0.0

    top = gain_vec[:k]
    discounts = 1.0 / np.log2(np.arange(2, top.size + 2, dtype=np.float64))
    dcg = float(np.sum(top * discounts))

    ideal_sorted = np.sort(ideal_pool)[::-1][:k]
    ideal_discounts = 1.0 / np.log2(
        np.arange(2, ideal_sorted.size + 2, dtype=np.float64)
    )
    idcg = float(np.sum(ideal_sorted * ideal_discounts))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Reciprocal Rank (RR) -- the per-query term of MRR
# ---------------------------------------------------------------------------
def reciprocal_rank(
    retrieved_labels: Optional[ArrayLike] = None,
    query_label: Optional[int] = None,
    k: Optional[int] = None,
    *,
    relevant_mask: Optional[ArrayLike] = None,
) -> float:
    r"""Reciprocal Rank -- ``1 / rank`` of the first relevant item.

    Formula
    -------
    .. math:: RR = \frac{1}{\text{rank of first relevant item}}

    (ranks are 1-based). If no relevant item appears in the list (optionally
    truncated to the first ``k`` ranks) the reciprocal rank is ``0.0``. The mean
    of this over queries is the Mean Reciprocal Rank (MRR).

    Returns
    -------
    float in ``[0, 1]``.
    """
    mask = _as_relevance_mask(retrieved_labels, query_label, relevant_mask)
    if k is not None:
        mask = mask[: _validate_k(k)]
    if mask.size == 0:
        return 0.0
    hits = np.flatnonzero(mask)
    if hits.size == 0:
        return 0.0
    first_rank = int(hits[0]) + 1  # 1-based
    return 1.0 / first_rank


# ---------------------------------------------------------------------------
# Vectorised batch helpers
# ---------------------------------------------------------------------------
def _coerce_2d_labels(retrieved_labels_2d: ArrayLike) -> np.ndarray:
    """Coerce a (Q, k) ranked-label matrix to a 2-D numpy array."""
    arr = np.asarray(retrieved_labels_2d)
    if arr.ndim != 2:
        raise ValueError(
            "retrieved_labels_2d must be 2-D (Q queries x K ranks); got shape "
            f"{arr.shape}"
        )
    return arr


def batch_f1_at_k(
    retrieved_labels_2d: ArrayLike,
    query_labels: ArrayLike,
    total_relevant_per_query: ArrayLike,
    k: int,
    mode: str = "raw",
) -> np.ndarray:
    r"""Vectorised F1@K over a batch of queries (class-equality relevance).

    Parameters
    ----------
    retrieved_labels_2d:
        ``(Q, R)`` integer array: row *q* holds the labels of the items
        retrieved for query *q*, in rank order (``R >= k`` recommended; if
        ``R < k`` the missing ranks count as non-relevant).
    query_labels:
        ``(Q,)`` integer array of per-query class labels.
    total_relevant_per_query:
        ``(Q,)`` integer array of ``R_q`` per query.
    k:
        Positive cutoff.
    mode:
        Recall convention, ``"raw"`` or ``"capped"`` (see :func:`recall_at_k`).

    Returns
    -------
    ``(Q,)`` float array of F1@K values, one per query. Uses the exact
    closed form :math:`F1@K = 2 r_K / (K + \text{denom})` with
    ``denom = R_q`` (raw) or ``min(K, R_q)`` (capped), and is ``0`` where
    ``R_q == 0``.
    """
    k = _validate_k(k)
    mode = _validate_mode(mode)
    labels = _coerce_2d_labels(retrieved_labels_2d)
    q_labels = np.asarray(query_labels)
    r_q = np.asarray(total_relevant_per_query).astype(np.float64)

    q = labels.shape[0]
    if q_labels.shape[0] != q or r_q.shape[0] != q:
        raise ValueError(
            "query_labels and total_relevant_per_query must have length "
            f"Q={q}; got {q_labels.shape[0]} and {r_q.shape[0]}"
        )

    # Boolean relevance over the top-k ranks, vectorised: (Q, min(k, R)).
    top = labels[:, :k]
    rel = top == q_labels[:, None]
    r_k = np.count_nonzero(rel, axis=1).astype(np.float64)  # (Q,)

    if mode == "raw":
        denom = k + r_q
    else:  # capped
        denom = k + np.minimum(k, r_q)

    out = np.zeros(q, dtype=np.float64)
    valid = r_q > 0
    # F1@K = 2 r_K / (K + denom_recall); guard against R_q == 0.
    out[valid] = (2.0 * r_k[valid]) / denom[valid]
    return out


def mean_metrics(
    retrieved_labels_2d: ArrayLike,
    query_labels: ArrayLike,
    total_relevant_per_query: ArrayLike,
    k: int,
    mode: str = "raw",
) -> dict[str, float]:
    r"""Macro-averaged retrieval metrics over a batch of queries.

    Computes the per-query metrics (class-equality relevance) and returns their
    **macro average** (simple mean over queries), matching how ``mR``/Recall@K
    are aggregated for retrieval and how PS-11 averages F1@K within an
    evaluation cell.

    Parameters are identical to :func:`batch_f1_at_k`.

    Returns
    -------
    dict with keys (``k`` is substituted, e.g. ``"P@5"``):

    * ``"P@{k}"``   -- mean Precision@K
    * ``"R@{k}"``   -- mean Recall@K (under ``mode``)
    * ``"F1@{k}"``  -- mean F1@K (under ``mode``)
    * ``"mAP"``     -- mean Average Precision (whole list)
    * ``"nDCG@{k}"``-- mean nDCG@K
    * ``"MRR"``     -- mean Reciprocal Rank
    * ``"n_queries"`` -- number of queries averaged (as float)

    Queries with ``R_q == 0`` are kept and contribute ``0`` to every metric
    (they are degenerate but counted, so the denominator is stable); callers
    who wish to drop them should filter beforehand.
    """
    k = _validate_k(k)
    mode = _validate_mode(mode)
    labels = _coerce_2d_labels(retrieved_labels_2d)
    q_labels = np.asarray(query_labels)
    r_q = np.asarray(total_relevant_per_query).astype(np.int64)
    q = labels.shape[0]

    if q == 0:
        return {
            f"P@{k}": 0.0,
            f"R@{k}": 0.0,
            f"F1@{k}": 0.0,
            "mAP": 0.0,
            f"nDCG@{k}": 0.0,
            "MRR": 0.0,
            "n_queries": 0.0,
        }

    # F1@K vectorised (also yields P@K cheaply via the same top-k relevance).
    f1_vec = batch_f1_at_k(labels, q_labels, r_q, k, mode)

    top = labels[:, :k]
    rel_top = top == q_labels[:, None]
    r_k = np.count_nonzero(rel_top, axis=1).astype(np.float64)
    p_vec = r_k / k
    if mode == "raw":
        r_denom = r_q.astype(np.float64)
    else:
        r_denom = np.minimum(k, r_q).astype(np.float64)
    r_vec = np.zeros(q, dtype=np.float64)
    nz = r_q > 0
    r_vec[nz] = r_k[nz] / r_denom[nz]

    # AP, nDCG, RR are computed per-query (cheap loops; Q is modest in eval).
    ap_vec = np.empty(q, dtype=np.float64)
    ndcg_vec = np.empty(q, dtype=np.float64)
    rr_vec = np.empty(q, dtype=np.float64)
    for i in range(q):
        row = labels[i]
        ql = q_labels[i]
        rq_i = int(r_q[i])
        ap_vec[i] = average_precision(row, ql, rq_i)
        ndcg_vec[i] = ndcg_at_k(row, ql, rq_i, k)
        rr_vec[i] = reciprocal_rank(row, ql)

    return {
        f"P@{k}": float(np.mean(p_vec)),
        f"R@{k}": float(np.mean(r_vec)),
        f"F1@{k}": float(np.mean(f1_vec)),
        "mAP": float(np.mean(ap_vec)),
        f"nDCG@{k}": float(np.mean(ndcg_vec)),
        "MRR": float(np.mean(rr_vec)),
        "n_queries": float(q),
    }
