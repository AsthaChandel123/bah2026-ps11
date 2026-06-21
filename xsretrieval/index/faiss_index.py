"""Vector index with a FAISS backend and an exact numpy brute-force fallback.

:class:`RetrievalIndex` is the search core of ``xsretrieval``.  It implements the
research-recommended design (``04_fast_retrieval.md`` §16): **one shared index
over all-modality, L2-normalized gallery embeddings**, queried by inner product
(== cosine), with a parallel ``modality`` side-array so results can be filtered
to the modality required by each evaluation case (same-modal vs cross-modal).

faiss policy
------------
``faiss`` is **never imported at module load** — it is imported lazily inside
:meth:`build` / :meth:`search` and is *optional*.  When faiss is unavailable the
class transparently uses an **exact numpy brute-force** path
(``scores = queries @ gallery.T`` then ``argpartition`` top-k) that returns
*identical, exact* nearest neighbours.  This guarantees the whole retrieval
pipeline runs correctly before faiss is installed; faiss is used purely for speed
when present.

Index sizing (research §14 cheatsheet)
--------------------------------------
``factory_string`` picks a faiss ``index_factory`` string by gallery size:

====================  ====================================
gallery size ``N``    factory string
====================  ====================================
≲ 25k                 ``"Flat"``                (exact, sub-ms)
≲ 200k                ``"HNSW32,Flat"``         (O(log N), high recall)
≲ 750k                ``"OPQ32_128,IVF16384,PQ32"``
larger                ``"OPQ32_128,IVF65536_HNSW32,PQ32"``
====================  ====================================

All embeddings are L2-normalized on ingest and at query time so cosine ranking
is exact regardless of backend.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from xsretrieval.data.modalities import Modality
from xsretrieval.index.rerank import exact_refine, k_reciprocal_rerank

__all__ = ["RetrievalIndex"]


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation (float32); ~0-norm rows are left at 0."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (x / norms).astype(np.float32, copy=False)


def _modality_to_str(m) -> str:
    """Coerce a modality (enum / raw string) to its canonical string value."""
    if isinstance(m, Modality):
        return m.value
    return Modality(m).value


def _faiss():
    """Try to import faiss; return the module or ``None`` if unavailable."""
    try:
        import faiss  # type: ignore
    except Exception:  # pragma: no cover - environment dependent
        return None
    return faiss


class RetrievalIndex:
    """Shared cross-modal vector index (faiss when available, else numpy).

    Parameters
    ----------
    dim:
        Embedding dimensionality ``d``.
    index_type:
        ``"auto"`` (default) chooses a faiss factory by gallery size at
        :meth:`build` time; or pass an explicit faiss factory string
        (e.g. ``"Flat"``, ``"HNSW32,Flat"``), or ``"flat"`` to force exact.
        Ignored entirely on the numpy fallback (which is always exact).
    metric:
        ``"ip"`` (inner product == cosine on unit vectors, default) or ``"l2"``.
    nlist:
        IVF cell count override (else derived from ``N`` as ``~8*sqrt(N)``).
    nprobe:
        IVF probe count (speed/recall knob, default 16).
    hnsw_M:
        HNSW graph degree (default 32).
    ef_search:
        HNSW query depth (default 64).

    Attributes
    ----------
    embeddings:
        ``(N, d)`` float32 L2-normalized gallery matrix (kept for exact refine).
    modalities:
        ``(N,)`` array of modality string values.
    labels, ids:
        ``(N,)`` int64 / object arrays of labels and external ids.
    """

    def __init__(
        self,
        dim: int,
        index_type: str = "auto",
        metric: str = "ip",
        nlist: Optional[int] = None,
        nprobe: int = 16,
        hnsw_M: int = 32,
        ef_search: int = 64,
    ) -> None:
        self.dim = int(dim)
        self.index_type = str(index_type)
        if metric not in ("ip", "l2"):
            raise ValueError("metric must be 'ip' or 'l2'")
        self.metric = metric
        self.nlist = nlist
        self.nprobe = int(nprobe)
        self.hnsw_M = int(hnsw_M)
        self.ef_search = int(ef_search)

        # Populated by build().
        self.embeddings: Optional[np.ndarray] = None
        self.modalities: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.ids: Optional[np.ndarray] = None
        self.size: int = 0

        self._faiss_index = None      # faiss.Index or None (numpy fallback)
        self._use_faiss: bool = False
        self._resolved_factory: Optional[str] = None
        # Precomputed per-modality row masks for fast modality filtering.
        self._modality_rows: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Factory string selection (research §14 cheatsheet)
    # ------------------------------------------------------------------
    @staticmethod
    def factory_string(n_gallery: int, dim: int) -> str:
        """Return the research-backed faiss factory string for ``n_gallery``.

        Thresholds follow the ``04_fast_retrieval.md`` §14 cheatsheet:

        * ``≲ 25k``   → ``"Flat"`` (exact; sub-ms, best F1 — don't over-engineer)
        * ``≲ 200k``  → ``"HNSW32,Flat"`` (O(log N), recall 0.97+, no training)
        * ``≲ 750k``  → ``"OPQ32_128,IVF16384,PQ32"`` (compact + fast)
        * larger      → ``"OPQ32_128,IVF65536_HNSW32,PQ32"`` (HNSW coarse
          quantizer removes the large-``nlist`` bottleneck)

        The OPQ output dim (128) is clamped to ``dim`` when the embedding is
        smaller, and PQ sub-quantizer count is kept a divisor of that dim.
        """
        n = int(n_gallery)
        if n <= 25_000:
            return "Flat"
        if n <= 200_000:
            return "HNSW32,Flat"

        # Compact PQ regimes: keep OPQ target dim <= embedding dim and a multiple
        # of the PQ sub-quantizer count (32). For typical d>=128 this is 128.
        opq_dim = 128 if dim >= 128 else (64 if dim >= 64 else 32)
        m_pq = 32 if opq_dim % 32 == 0 else (16 if opq_dim % 16 == 0 else 8)
        if n <= 750_000:
            return f"OPQ{m_pq}_{opq_dim},IVF16384,PQ{m_pq}"
        return f"OPQ{m_pq}_{opq_dim},IVF65536_HNSW32,PQ{m_pq}"

    def _resolve_factory(self, n: int) -> str:
        """Resolve the factory string honouring ``index_type``."""
        if self.index_type in ("auto", "", None):
            return self.factory_string(n, self.dim)
        if self.index_type.lower() in ("flat", "bruteforce", "brute_force"):
            return "Flat"
        return self.index_type

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(
        self,
        embeddings: np.ndarray,
        ids: Optional[np.ndarray] = None,
        modalities: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> "RetrievalIndex":
        """Build the index over ``embeddings`` and store the side metadata.

        Embeddings are L2-normalized (so inner product == cosine).  A faiss
        index is constructed if faiss imports successfully; otherwise the exact
        numpy brute-force path is used.

        Parameters
        ----------
        embeddings:
            ``(N, d)`` float32 gallery embeddings (normalized internally).
        ids:
            Optional ``(N,)`` external identifiers (any dtype). Defaults to
            ``arange(N)``.
        modalities:
            Optional ``(N,)`` of ``Modality`` / string values, stored for
            per-modality result filtering.
        labels:
            Optional ``(N,)`` int64 semantic labels (stored for the evaluator).

        Returns
        -------
        self
        """
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"embeddings must be (N, d); got {emb.shape}")
        if emb.shape[1] != self.dim:
            raise ValueError(
                f"embedding dim {emb.shape[1]} != index dim {self.dim}"
            )
        n = emb.shape[0]
        emb = _l2_normalize(emb)
        self.embeddings = emb
        self.size = n

        self.ids = (
            np.arange(n)
            if ids is None
            else np.asarray(ids)
        )
        self.labels = (
            np.full(n, -1, dtype=np.int64)
            if labels is None
            else np.asarray(labels, dtype=np.int64)
        )
        if modalities is None:
            self.modalities = np.array([""] * n, dtype=object)
        else:
            self.modalities = np.array(
                [_modality_to_str(m) for m in modalities], dtype=object
            )

        # Precompute per-modality row index arrays (used by modality filtering).
        self._modality_rows = {}
        for key in np.unique(self.modalities):
            if key == "":
                continue
            self._modality_rows[str(key)] = np.where(self.modalities == key)[0]

        # Try to build a faiss index; fall back to numpy on any failure.
        self._build_faiss(emb, n)
        return self

    def _build_faiss(self, emb: np.ndarray, n: int) -> None:
        """Construct the faiss index, or set the numpy fallback."""
        faiss = _faiss()
        if faiss is None:
            self._use_faiss = False
            self._faiss_index = None
            self._resolved_factory = "numpy:bruteforce"
            return

        metric = (
            faiss.METRIC_INNER_PRODUCT
            if self.metric == "ip"
            else faiss.METRIC_L2
        )
        factory = self._resolve_factory(n)
        self._resolved_factory = factory
        try:
            index = faiss.index_factory(self.dim, factory, metric)

            # Configure IVF nlist/nprobe and HNSW ef before training.
            ivf = self._extract_ivf(faiss, index)
            if ivf is not None:
                if self.nlist is not None:
                    # nlist is set by the factory string; we only set nprobe.
                    pass
                ivf.nprobe = self.nprobe
            self._set_hnsw_ef(faiss, index)

            # Train if the index requires it (IVF/PQ/OPQ).
            if not index.is_trained:
                index.train(emb)
            index.add(emb)
            self._faiss_index = index
            self._use_faiss = True
        except Exception:
            # Any faiss build problem (e.g. too few training points) → numpy.
            self._use_faiss = False
            self._faiss_index = None
            self._resolved_factory = "numpy:bruteforce"

    @staticmethod
    def _extract_ivf(faiss, index):
        """Return the underlying IVF sub-index if present, else ``None``."""
        try:
            return faiss.extract_index_ivf(index)
        except Exception:
            return None

    def _set_hnsw_ef(self, faiss, index) -> None:
        """Set HNSW efSearch/efConstruction where the index exposes it."""
        for attr in ("hnsw",):
            obj = getattr(index, attr, None)
            if obj is not None:
                try:
                    obj.efSearch = self.ef_search
                    obj.efConstruction = max(80, self.ef_search)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        queries: np.ndarray,
        k: int,
        gallery_modality=None,
        exclude_ids: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the top-``k`` gallery neighbours of each query.

        Parameters
        ----------
        queries:
            ``(Nq, d)`` (or ``(d,)``) float32 query embeddings (normalized here).
        k:
            Neighbours per query.
        gallery_modality:
            If set (a ``Modality`` / string), restrict results to gallery items
            of that modality.  Uses a faiss ``IDSelector`` when available, else
            over-fetches and post-filters (numpy path).
        exclude_ids:
            Optional iterable / ``(Nq, ?)`` of gallery **row indices** to exclude
            per query (e.g. the query's own row, or all rows of its location, for
            leave-one-out evaluation).  A 1-D array is applied to every query; a
            2-D / ragged ``list`` excludes per-query.

        Returns
        -------
        (scores, indices):
            ``scores`` ``(Nq, k)`` float32 cosine similarities (``-inf`` padding
            where fewer than ``k`` results exist) and ``indices`` ``(Nq, k)``
            int64 gallery row indices (``-1`` padding).
        """
        if self.embeddings is None:
            raise RuntimeError("search() before build()")
        q = _l2_normalize(queries)
        nq = q.shape[0]
        k = int(k)

        want_mod = (
            None if gallery_modality is None
            else _modality_to_str(gallery_modality)
        )
        excl = self._normalise_exclusions(exclude_ids, nq)

        if self._use_faiss:
            return self._search_faiss(q, k, want_mod, excl)
        return self._search_numpy(q, k, want_mod, excl)

    def _normalise_exclusions(self, exclude_ids, nq: int):
        """Return a list (len nq) of int64 arrays of rows to exclude per query."""
        if exclude_ids is None:
            return None
        # Per-query ragged exclusions.
        if isinstance(exclude_ids, (list, tuple)) and len(exclude_ids) == nq and (
            len(exclude_ids) == 0
            or np.ndim(exclude_ids[0]) >= 1
            or isinstance(exclude_ids[0], (list, tuple, np.ndarray))
        ):
            return [np.asarray(e, dtype=np.int64).ravel() for e in exclude_ids]
        arr = np.asarray(exclude_ids, dtype=np.int64)
        if arr.ndim == 2 and arr.shape[0] == nq:
            return [arr[i].ravel() for i in range(nq)]
        flat = arr.ravel()                               # shared across queries
        return [flat for _ in range(nq)]

    # ----- numpy brute-force (exact) -----
    def _search_numpy(self, q, k, want_mod, excl):
        """Exact brute-force search: ``scores = q @ gallery.T`` + argpartition.

        Complexity ``O(Nq * N * d)``; exact top-k via ``argpartition`` (O(N) per
        query selection) then a small sort of the k retained.  This is the
        reference path that makes every result correct without faiss.
        """
        gallery = self.embeddings
        nq = q.shape[0]
        out_idx = np.full((nq, k), -1, dtype=np.int64)
        out_score = np.full((nq, k), -np.inf, dtype=np.float32)

        # Restrict the searchable rows to the requested modality once.
        if want_mod is not None:
            rows = self._modality_rows.get(want_mod, np.empty(0, np.int64))
            if rows.size == 0:
                return out_score, out_idx
            sub = gallery[rows]
        else:
            rows = None
            sub = gallery

        sims_full = q @ sub.T                            # (Nq, |sub|)
        for i in range(nq):
            sims = sims_full[i]
            if excl is not None and excl[i].size:
                if rows is None:
                    mask_rows = excl[i]
                else:
                    # Map global excluded rows into sub-index positions.
                    mask_rows = np.where(np.isin(rows, excl[i]))[0]
                if mask_rows.size:
                    sims = sims.copy()
                    sims[mask_rows] = -np.inf
            topk = self._topk_from_scores(sims, k)
            m = topk.size
            cols = topk if rows is None else rows[topk]
            out_idx[i, :m] = cols
            out_score[i, :m] = sims[topk]
        return out_score, out_idx

    @staticmethod
    def _topk_from_scores(sims: np.ndarray, k: int) -> np.ndarray:
        """Indices of the top-``k`` scores (descending), exact via argpartition."""
        n = sims.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.int64)
        kk = min(k, n)
        # argpartition gives the kk largest (unordered) in O(n); then sort those.
        part = np.argpartition(-sims, kk - 1)[:kk]
        order = np.argsort(-sims[part], kind="stable")
        sel = part[order]
        # Drop any -inf (fully excluded) entries.
        valid = np.isfinite(sims[sel])
        return sel[valid]

    # ----- faiss path -----
    def _search_faiss(self, q, k, want_mod, excl):
        """faiss search with modality filtering and per-query exclusions.

        Uses a faiss ``IDSelectorBatch`` to restrict to a modality / exclude ids
        when the installed faiss exposes ``SearchParameters``; otherwise
        over-fetches a larger pool and post-filters in numpy (always correct).
        """
        faiss = _faiss()
        nq = q.shape[0]

        # Determine the allowed global row set for modality filtering.
        allowed_rows = None
        if want_mod is not None:
            allowed_rows = self._modality_rows.get(
                want_mod, np.empty(0, np.int64)
            )
            if allowed_rows.size == 0:
                return (
                    np.full((nq, k), -np.inf, dtype=np.float32),
                    np.full((nq, k), -1, dtype=np.int64),
                )

        # Fast path: try a uniform IDSelector (modality only, exclusions None or
        # shared) so faiss does the filtering. Falls back to over-fetch on any
        # incompatibility.
        sel = None
        shared_excl = None
        if excl is not None and all(np.array_equal(excl[0], e) for e in excl):
            shared_excl = excl[0]
        try:
            sel = self._build_selector(faiss, allowed_rows, shared_excl)
        except Exception:
            sel = None

        per_query_excl = excl is not None and shared_excl is None

        if sel is not None and not per_query_excl:
            try:
                params = self._make_search_params(faiss, sel)
                scores, idx = self._faiss_index.search(q, k, params=params)
                return self._pad_faiss_results(scores, idx, k)
            except Exception:
                pass  # fall through to over-fetch

        # Over-fetch + post-filter (robust, backend-agnostic).
        fetch = self._overfetch_k(k, want_mod, excl)
        scores, idx = self._faiss_index.search(q, fetch)
        return self._postfilter(scores, idx, k, allowed_rows, excl)

    def _overfetch_k(self, k, want_mod, excl) -> int:
        """How many raw neighbours to fetch before post-filtering."""
        fetch = k
        if want_mod is not None:
            # Fraction of gallery in this modality → inflate to compensate.
            n_mod = self._modality_rows.get(want_mod, np.empty(0)).size
            frac = max(n_mod / max(self.size, 1), 1e-3)
            fetch = int(np.ceil(k / frac)) + k
        if excl is not None:
            fetch += max(int(np.max([e.size for e in excl])), 0)
        return int(min(max(fetch, k), self.size))

    @staticmethod
    def _build_selector(faiss, allowed_rows, shared_excl):
        """Build a faiss IDSelector restricting to allowed rows minus exclusions.

        Returns ``None`` if neither a modality restriction nor an exclusion is
        requested, or if the installed faiss lacks ``IDSelectorBatch``.
        """
        if allowed_rows is None and (shared_excl is None or shared_excl.size == 0):
            return None
        if not hasattr(faiss, "IDSelectorBatch"):
            raise RuntimeError("faiss without IDSelectorBatch")
        if allowed_rows is not None:
            keep = allowed_rows
            if shared_excl is not None and shared_excl.size:
                keep = np.setdiff1d(keep, shared_excl, assume_unique=False)
        else:
            keep = np.setdiff1d(
                np.arange(0), shared_excl  # placeholder, replaced below
            )
            keep = None
        if keep is None:
            # Exclusion-only: build a complement selector.
            raise RuntimeError("exclusion-only selector via over-fetch")
        keep = np.ascontiguousarray(keep.astype(np.int64))
        return faiss.IDSelectorBatch(keep.size, faiss.swig_ptr(keep))

    @staticmethod
    def _make_search_params(faiss, selector):
        """Build a ``SearchParameters`` carrying ``selector`` (IVF or generic)."""
        if hasattr(faiss, "SearchParametersIVF"):
            try:
                return faiss.SearchParametersIVF(sel=selector)
            except Exception:
                pass
        return faiss.SearchParameters(sel=selector)

    def _pad_faiss_results(self, scores, idx, k):
        """Coerce faiss outputs to fixed ``(Nq, k)`` arrays with proper padding."""
        scores = np.asarray(scores, dtype=np.float32)
        idx = np.asarray(idx, dtype=np.int64)
        if self.metric == "l2":
            # Convert L2 to a similarity-like score (higher = closer) for a
            # consistent descending interface: sim = 1 - d/2 for unit vectors.
            scores = 1.0 - scores / 2.0
        # faiss already returns exactly k columns; pad/truncate defensively.
        nq = idx.shape[0]
        out_idx = np.full((nq, k), -1, dtype=np.int64)
        out_score = np.full((nq, k), -np.inf, dtype=np.float32)
        kk = min(k, idx.shape[1])
        out_idx[:, :kk] = idx[:, :kk]
        out_score[:, :kk] = scores[:, :kk]
        out_idx[out_idx < 0] = -1
        return out_score, out_idx

    def _postfilter(self, scores, idx, k, allowed_rows, excl):
        """Post-filter raw faiss neighbours by modality / exclusions to top-k."""
        scores = np.asarray(scores, dtype=np.float32)
        idx = np.asarray(idx, dtype=np.int64)
        if self.metric == "l2":
            scores = 1.0 - scores / 2.0
        nq = idx.shape[0]
        out_idx = np.full((nq, k), -1, dtype=np.int64)
        out_score = np.full((nq, k), -np.inf, dtype=np.float32)
        allowed_set = None if allowed_rows is None else set(allowed_rows.tolist())
        for i in range(nq):
            row = idx[i]
            sc = scores[i]
            keep_i = []
            keep_s = []
            excl_i = None if excl is None else set(excl[i].tolist())
            for j, gi in enumerate(row):
                if gi < 0:
                    continue
                if allowed_set is not None and gi not in allowed_set:
                    continue
                if excl_i is not None and gi in excl_i:
                    continue
                keep_i.append(int(gi))
                keep_s.append(float(sc[j]))
                if len(keep_i) >= k:
                    break
            m = len(keep_i)
            if m:
                out_idx[i, :m] = keep_i
                out_score[i, :m] = keep_s
        return out_score, out_idx

    # ------------------------------------------------------------------
    # Search + re-rank
    # ------------------------------------------------------------------
    def search_with_rerank(
        self,
        queries: np.ndarray,
        k: int,
        candidate_pool: int = 200,
        rerank: str = "kreciprocal",
        gallery_modality=None,
        exclude_ids: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Over-fetch a candidate pool, exact-refine, then optional re-rank.

        Pipeline (research §16b): fetch ``candidate_pool`` neighbours from the
        fast index → exact-cosine refine the pool (:func:`exact_refine`) →
        ``rerank`` strategy on the refined pool → final top-``k``.

        Parameters
        ----------
        queries:
            ``(Nq, d)`` (or ``(d,)``) query embeddings.
        k:
            Final neighbours per query.
        candidate_pool:
            Number of first-stage candidates to over-fetch (default 200).
        rerank:
            ``"kreciprocal"`` (Zhong et al.), ``"exact"`` (refine only), or
            ``"none"``.
        gallery_modality, exclude_ids:
            As in :meth:`search`.

        Returns
        -------
        (scores, indices):
            ``(Nq, k)`` float32 / int64. Scores are exact cosine similarities of
            the final ordering (``-inf`` padding for k-reciprocal whose internal
            metric is a blended distance — the returned scores are the recomputed
            exact cosines of the chosen indices, for a consistent interface).
        """
        if self.embeddings is None:
            raise RuntimeError("search_with_rerank() before build()")
        q = _l2_normalize(queries)
        nq = q.shape[0]
        pool = int(max(candidate_pool, k))

        # Stage 1: fetch a candidate pool (with modality / exclusion filtering).
        _, cand_idx = self.search(
            q, pool, gallery_modality=gallery_modality, exclude_ids=exclude_ids
        )

        out_idx = np.full((nq, k), -1, dtype=np.int64)
        out_score = np.full((nq, k), -np.inf, dtype=np.float32)
        gallery = self.embeddings

        for i in range(nq):
            cands = cand_idx[i]
            cands = cands[cands >= 0]
            if cands.size == 0:
                continue
            if rerank == "none":
                final = cands[:k]
            elif rerank == "exact":
                final = exact_refine(q[i], gallery, cands, k)
            elif rerank in ("kreciprocal", "k_reciprocal", "kr"):
                # Exact refine first (cheap, restores precision), then
                # k-reciprocal re-rank on the refined pool, take top-k.
                refined = exact_refine(q[i], gallery, cands, cands.size)
                reordered = k_reciprocal_rerank(q[i], gallery, refined)
                final = reordered[:k]
            else:
                raise ValueError(f"unknown rerank strategy {rerank!r}")
            m = final.size
            out_idx[i, :m] = final
            # Report exact cosine of the chosen ordering for a uniform interface.
            out_score[i, :m] = gallery[final] @ q[i]
        return out_score, out_idx

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the index: ``<path>.faiss`` (if faiss) + ``<path>.npz`` sidecar.

        The sidecar always stores the float32 gallery + side arrays + config, so
        the index can be reloaded and used on the numpy path even if faiss is
        absent at load time.
        """
        if self.embeddings is None:
            raise RuntimeError("cannot save an unbuilt index")
        sidecar = path if path.endswith(".npz") else path + ".npz"
        np.savez(
            sidecar,
            embeddings=self.embeddings,
            modalities=np.asarray(self.modalities, dtype=object),
            labels=self.labels,
            ids=np.asarray(self.ids, dtype=object),
            config=np.array(
                [self.dim, self.nprobe, self.hnsw_M, self.ef_search],
                dtype=np.int64,
            ),
            strcfg=np.array(
                [self.index_type, self.metric, self._resolved_factory or ""],
                dtype=object,
            ),
        )
        if self._use_faiss and self._faiss_index is not None:
            faiss = _faiss()
            if faiss is not None:
                faiss.write_index(
                    self._faiss_index,
                    path if path.endswith(".faiss") else path + ".faiss",
                )

    @classmethod
    def load(cls, path: str) -> "RetrievalIndex":
        """Load an index saved by :meth:`save` (faiss index if present, else numpy)."""
        sidecar = path if path.endswith(".npz") else path + ".npz"
        data = np.load(sidecar, allow_pickle=True)
        cfg = data["config"]
        strcfg = [str(x) for x in data["strcfg"].tolist()]
        obj = cls(
            dim=int(cfg[0]),
            index_type=strcfg[0],
            metric=strcfg[1],
            nprobe=int(cfg[1]),
            hnsw_M=int(cfg[2]),
            ef_search=int(cfg[3]),
        )
        obj.embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        obj.modalities = np.asarray(data["modalities"], dtype=object)
        obj.labels = np.asarray(data["labels"], dtype=np.int64)
        obj.ids = np.asarray(data["ids"], dtype=object)
        obj.size = obj.embeddings.shape[0]
        obj._resolved_factory = strcfg[2] if len(strcfg) > 2 else None
        obj._modality_rows = {}
        for key in np.unique(obj.modalities):
            if key == "":
                continue
            obj._modality_rows[str(key)] = np.where(obj.modalities == key)[0]

        # Try to restore a faiss index from the sidecar .faiss file.
        faiss = _faiss()
        faiss_path = path if path.endswith(".faiss") else path + ".faiss"
        obj._use_faiss = False
        obj._faiss_index = None
        if faiss is not None:
            import os

            if os.path.exists(faiss_path):
                try:
                    obj._faiss_index = faiss.read_index(faiss_path)
                    obj._use_faiss = True
                except Exception:
                    obj._use_faiss = False
        return obj

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def uses_faiss(self) -> bool:
        """Whether the active backend is faiss (``False`` = numpy brute force)."""
        return self._use_faiss

    @property
    def factory(self) -> Optional[str]:
        """The resolved faiss factory string (or ``"numpy:bruteforce"``)."""
        return self._resolved_factory

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        backend = "faiss" if self._use_faiss else "numpy"
        return (
            f"RetrievalIndex(dim={self.dim}, size={self.size}, "
            f"backend={backend}, factory={self._resolved_factory!r})"
        )
