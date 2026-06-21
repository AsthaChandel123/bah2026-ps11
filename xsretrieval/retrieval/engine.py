"""End-to-end retrieval engine: encode → whiten → index → query.

:class:`RetrievalEngine` ties the pieces together into the inference / evaluation
API used by the rest of ``xsretrieval`` (research ``04_fast_retrieval.md`` §16,
``03_crossmodal_alignment.md`` §14):

    samples ─► backbone.embed ─► projection ─► per-modality whitening ─► L2-norm
            ─► RetrievalIndex (shared cross-modal, faiss-or-numpy) ─► top-k

It is deliberately **duck-typed** and CPU-fast.  The components are injected:

* ``backbone``   — any object with ``.embed(images, modality) -> (B, D)`` (the
  models team's frozen foundation backbone).  ``images`` is a numpy ``(B,C,H,W)``
  batch; the return is a numpy / array-like ``(B, D)``.
* ``projection`` — optional; an object with ``.forward`` or ``__call__`` mapping
  ``(B, D) -> (B, D')`` (the shared projection head).
* ``whitener``   — optional :class:`PerModalityWhitener` / :class:`GlobalWhitener`
  (the modality-gap fix; ``.transform(emb, modality) -> emb``).

No torch / faiss is imported at module level; whatever the backbone/projection
use is their concern (and they are only touched inside :meth:`encode`).  With the
numpy brute-force index the entire query path is exact and dependency-light.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from xsretrieval.data.modalities import Modality
from xsretrieval.index.faiss_index import RetrievalIndex

__all__ = ["RetrievalEngine"]


def _to_numpy(x: Any) -> np.ndarray:
    """Best-effort conversion of an array-like / torch tensor to numpy float32.

    Avoids importing torch: relies on the duck-typed ``.detach()/.cpu()/.numpy()``
    protocol when present, else ``np.asarray``.
    """
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    # torch.Tensor (or similar) without importing torch.
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation (float32)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    norms = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)
    return (x / norms).astype(np.float32, copy=False)


def _modality_value(m) -> str:
    """Canonical string value of a modality (enum or raw string)."""
    return m.value if isinstance(m, Modality) else Modality(m).value


def _is_sample(obj: Any) -> bool:
    """Duck-typed check for a ``Sample`` (has image + modality attributes)."""
    return hasattr(obj, "image") and hasattr(obj, "modality")


class RetrievalEngine:
    """Encode galleries/queries and run fast cross-modal retrieval.

    Parameters
    ----------
    backbone:
        Object exposing ``embed(images, modality) -> (B, D)`` (duck-typed).
    projection:
        Optional projection head (``.forward`` / ``__call__``: ``(B,D)->(B,D')``).
    whitener:
        Optional fitted whitener (``.transform(emb, modality)->emb``).  The
        single most important cross-modal accuracy lever (research §13).
    index_cfg:
        Optional dict of :class:`RetrievalIndex` kwargs (``index_type``,
        ``metric``, ``nprobe`` …).  Index ``dim`` is inferred from the encoded
        embeddings.
    rerank:
        Default re-ranking behaviour for :meth:`query` / :meth:`batch_query`.
        ``True``/``"kreciprocal"`` enables k-reciprocal re-ranking; ``False``
        / ``None`` uses plain search.  Can be overridden per call.
    """

    def __init__(
        self,
        backbone: Any,
        projection: Any = None,
        whitener: Any = None,
        index_cfg: Optional[dict] = None,
        rerank: bool = False,
    ) -> None:
        self.backbone = backbone
        self.projection = projection
        self.whitener = whitener
        self.index_cfg = dict(index_cfg) if index_cfg else {}
        self.rerank_default = rerank

        # Gallery state, populated by index_gallery().
        self.index: Optional[RetrievalIndex] = None
        self.gallery_ids: Optional[np.ndarray] = None
        self.gallery_labels: Optional[np.ndarray] = None
        self.gallery_modalities: Optional[np.ndarray] = None
        self.gallery_location_ids: Optional[np.ndarray] = None
        self.embed_dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def _apply_projection(self, emb: np.ndarray) -> np.ndarray:
        """Run the optional projection head on a ``(B, D)`` embedding batch."""
        if self.projection is None:
            return emb
        proj = self.projection
        if hasattr(proj, "forward"):
            out = proj.forward(emb)
        elif callable(proj):
            out = proj(emb)
        else:  # pragma: no cover - misconfiguration
            raise TypeError("projection must have .forward or be callable")
        return _to_numpy(out)

    def encode(
        self,
        samples_or_images: Any,
        modality=None,
        batch_size: int = 64,
    ) -> np.ndarray:
        """Encode samples / raw images to L2-normalized embeddings ``(N, D)``.

        Runs ``backbone.embed`` → optional projection → optional per-modality
        whitening → L2-normalisation.  Two input forms are accepted:

        * a sequence of ``Sample`` objects (each carries its own ``modality``);
          samples are grouped by modality so each ``backbone.embed`` call is
          modality-homogeneous and the whitener can be applied per modality.
        * a raw image batch ``(N, C, H, W)`` (or list of ``(C,H,W)``) **plus** an
          explicit ``modality`` argument applied to all of them.

        Parameters
        ----------
        samples_or_images:
            ``Sample`` list, or a numpy ``(N,C,H,W)`` array / list of images.
        modality:
            Required when passing raw images; ignored for ``Sample`` inputs.
        batch_size:
            Backbone mini-batch size.

        Returns
        -------
        np.ndarray
            ``(N, D)`` float32 L2-normalized embeddings, in input order.
        """
        items = list(samples_or_images) if _is_sequence(samples_or_images) else \
            self._split_array(samples_or_images)

        if len(items) == 0:
            d = self.embed_dim if self.embed_dim is not None else 1
            return np.zeros((0, d), dtype=np.float32)

        if _is_sample(items[0]):
            return self._encode_samples(items, batch_size)

        if modality is None:
            raise ValueError(
                "encode() needs an explicit `modality` for raw image inputs"
            )
        images = self._stack_images(items)
        emb = self._encode_image_batch(images, modality, batch_size)
        self.embed_dim = emb.shape[1]
        return emb

    def _encode_samples(self, samples: list, batch_size: int) -> np.ndarray:
        """Encode a list of ``Sample`` objects, grouping by modality."""
        n = len(samples)
        # Group indices by modality so each backbone call is homogeneous.
        groups: dict[str, list[int]] = {}
        for i, s in enumerate(samples):
            groups.setdefault(_modality_value(s.modality), []).append(i)

        out: Optional[np.ndarray] = None
        for mod_value, idxs in groups.items():
            imgs = self._stack_images([samples[i].image for i in idxs])
            emb = self._encode_image_batch(imgs, mod_value, batch_size)
            if out is None:
                out = np.zeros((n, emb.shape[1]), dtype=np.float32)
                self.embed_dim = emb.shape[1]
            out[idxs] = emb
        assert out is not None
        return out

    def _encode_image_batch(
        self, images: np.ndarray, modality, batch_size: int
    ) -> np.ndarray:
        """Backbone → projection → whitening → L2-norm for one modality batch."""
        mod_value = _modality_value(modality)
        mod_enum = Modality(mod_value)
        embs: list[np.ndarray] = []
        n = images.shape[0]
        bs = max(1, int(batch_size))
        for start in range(0, n, bs):
            chunk = images[start:start + bs]
            raw = self.backbone.embed(chunk, mod_enum)
            embs.append(_to_numpy(raw))
        emb = np.concatenate(embs, axis=0) if embs else np.zeros(
            (0, 1), dtype=np.float32
        )
        emb = self._apply_projection(emb)
        if self.whitener is not None:
            emb = _to_numpy(self.whitener.transform(emb, mod_enum))
        else:
            emb = _l2_normalize(emb)
        return emb.astype(np.float32, copy=False)

    @staticmethod
    def _split_array(arr: Any) -> list:
        """Turn a ``(N, C, H, W)`` array into a list of ``(C,H,W)`` images."""
        a = np.asarray(arr)
        if a.ndim == 4:
            return [a[i] for i in range(a.shape[0])]
        if a.ndim == 3:
            return [a]
        raise ValueError(
            f"raw image input must be (N,C,H,W) or (C,H,W); got shape {a.shape}"
        )

    @staticmethod
    def _stack_images(images: list) -> np.ndarray:
        """Stack a list of ``(C,H,W)`` arrays into a ``(N,C,H,W)`` float32 batch."""
        return np.stack([np.asarray(im, dtype=np.float32) for im in images], axis=0)

    # ------------------------------------------------------------------
    # Whitener fitting convenience
    # ------------------------------------------------------------------
    def fit_whitener(self, samples: list, whitener: Any = None) -> Any:
        """Fit (and attach) a per-modality whitener on ``samples``.

        Encodes ``samples`` **without** the current whitener (raw backbone +
        projection + L2-norm), groups the embeddings by modality, and fits the
        provided ``whitener`` (or the engine's existing one).  The fitted
        whitener is stored on the engine and returned.

        Parameters
        ----------
        samples:
            ``Sample`` list spanning the modalities to calibrate.
        whitener:
            Whitener instance to fit; defaults to ``self.whitener``.  Must expose
            ``fit(emb_by_mod)`` (per-modality) — a :class:`PerModalityWhitener`.

        Returns
        -------
        The fitted whitener.
        """
        target = whitener if whitener is not None else self.whitener
        if target is None:
            raise ValueError("no whitener provided to fit_whitener")

        # Encode raw (bypass any attached whitener) so the fit sees pre-whitened
        # backbone features.
        saved = self.whitener
        self.whitener = None
        try:
            emb = self.encode(samples)
        finally:
            self.whitener = saved

        mods = [_modality_value(s.modality) for s in samples]
        emb_by_mod: dict = {}
        keys = np.array(mods)
        for key in np.unique(keys):
            emb_by_mod[Modality(key)] = emb[keys == key]
        target.fit(emb_by_mod)
        self.whitener = target
        return target

    # ------------------------------------------------------------------
    # Gallery indexing
    # ------------------------------------------------------------------
    def index_gallery(self, samples: list) -> RetrievalIndex:
        """Encode all ``samples`` and build the shared cross-modal index.

        Stores gallery metadata (ids, labels, modalities, location_ids) for the
        evaluator and for leave-one-out exclusion.

        Parameters
        ----------
        samples:
            ``Sample`` list forming the gallery.

        Returns
        -------
        RetrievalIndex
            The built index (also stored on ``self.index``).
        """
        samples = list(samples)
        if not samples:
            raise ValueError("index_gallery received no samples")
        emb = self.encode(samples)
        n, d = emb.shape

        ids = np.array([s.id for s in samples], dtype=object)
        labels = np.array(
            [int(getattr(s, "label", -1)) for s in samples], dtype=np.int64
        )
        modalities = np.array(
            [_modality_value(s.modality) for s in samples], dtype=object
        )
        location_ids = np.array(
            [
                "" if getattr(s, "location_id", None) is None
                else str(s.location_id)
                for s in samples
            ],
            dtype=object,
        )

        index = RetrievalIndex(dim=d, **self.index_cfg)
        index.build(emb, ids=ids, modalities=modalities, labels=labels)

        self.index = index
        self.gallery_ids = ids
        self.gallery_labels = labels
        self.gallery_modalities = modalities
        self.gallery_location_ids = location_ids
        self.embed_dim = d
        return index

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def _encode_query(self, sample_or_emb: Any) -> np.ndarray:
        """Return a ``(1, D)`` embedding for a Sample / raw image / embedding."""
        if _is_sample(sample_or_emb):
            return self.encode([sample_or_emb])
        arr = np.asarray(sample_or_emb)
        if arr.ndim == 1 and (
            self.embed_dim is not None and arr.shape[0] == self.embed_dim
        ):
            # Already an embedding vector.
            return _l2_normalize(arr)
        if arr.ndim == 2 and (
            self.embed_dim is not None and arr.shape[1] == self.embed_dim
        ):
            return _l2_normalize(arr)
        raise TypeError(
            "query must be a Sample or a (D,)/(N,D) embedding matching the "
            "gallery dim; raw images must be encoded via encode(images, modality)"
        )

    def _resolve_rerank(self, rerank) -> Optional[str]:
        """Map the rerank flag/string to a strategy name (or ``None``)."""
        flag = self.rerank_default if rerank is None else rerank
        if flag in (False, None, "none", "off"):
            return None
        if flag is True:
            return "kreciprocal"
        return str(flag)

    def _exclusion_rows(
        self, query_sample: Any, exclude_same_location: bool
    ) -> Optional[np.ndarray]:
        """Gallery rows to exclude for a query (its own id, or whole location)."""
        if self.index is None:
            return None
        rows: list[int] = []
        # Exclude the query's own gallery row (id match), for leave-one-out.
        qid = getattr(query_sample, "id", None)
        if qid is not None and self.gallery_ids is not None:
            same_id = np.where(self.gallery_ids == qid)[0]
            rows.extend(int(r) for r in same_id)
        # Exclude all gallery rows from the same location.
        if exclude_same_location and self.gallery_location_ids is not None:
            qloc = getattr(query_sample, "location_id", None)
            if qloc is not None:
                same_loc = np.where(self.gallery_location_ids == str(qloc))[0]
                rows.extend(int(r) for r in same_loc)
        if not rows:
            return None
        return np.unique(np.asarray(rows, dtype=np.int64))

    def _format_hits(
        self, scores_row: np.ndarray, idx_row: np.ndarray
    ) -> list[dict]:
        """Build the list-of-dict result for one query from score/index rows."""
        hits: list[dict] = []
        rank = 0
        for sc, gi in zip(scores_row, idx_row):
            if gi < 0:
                continue
            gi = int(gi)
            hits.append(
                {
                    "id": (
                        self.gallery_ids[gi]
                        if self.gallery_ids is not None else gi
                    ),
                    "label": (
                        int(self.gallery_labels[gi])
                        if self.gallery_labels is not None else -1
                    ),
                    "modality": (
                        self.gallery_modalities[gi]
                        if self.gallery_modalities is not None else ""
                    ),
                    "score": float(sc),
                    "rank": rank,
                    "gallery_index": gi,
                }
            )
            rank += 1
        return hits

    def query(
        self,
        sample_or_emb: Any,
        k: int = 10,
        gallery_modality=None,
        exclude_same_location: bool = True,
        rerank=None,
    ) -> list[dict]:
        """Retrieve the top-``k`` gallery items for a single query.

        Parameters
        ----------
        sample_or_emb:
            A ``Sample``, or a ``(D,)`` / ``(1, D)`` embedding already in the
            gallery space.
        k:
            Number of results.
        gallery_modality:
            Restrict results to this modality (for same-/cross-modal cases).
        exclude_same_location:
            Drop gallery items sharing the query's ``location_id`` (and the
            query's own id) — the standard leave-one-out protocol so a query
            never retrieves itself / its co-registered twin.
        rerank:
            Override the engine default (``True`` → k-reciprocal, ``"exact"``,
            ``False`` → none).

        Returns
        -------
        list[dict]
            Up to ``k`` dicts ``{id, label, modality, score, rank,
            gallery_index}`` ordered best-first.
        """
        if self.index is None:
            raise RuntimeError("query() before index_gallery()")
        q = self._encode_query(sample_or_emb)
        excl = None
        if _is_sample(sample_or_emb):
            excl_rows = self._exclusion_rows(sample_or_emb, exclude_same_location)
            excl = None if excl_rows is None else [excl_rows]

        strategy = self._resolve_rerank(rerank)
        if strategy is None:
            scores, idx = self.index.search(
                q, k, gallery_modality=gallery_modality, exclude_ids=excl
            )
        else:
            scores, idx = self.index.search_with_rerank(
                q,
                k,
                rerank=strategy,
                gallery_modality=gallery_modality,
                exclude_ids=excl,
            )
        return self._format_hits(scores[0], idx[0])

    def batch_query(
        self,
        samples: list,
        k: int = 10,
        gallery_modality=None,
        exclude_same_location: bool = True,
        rerank=None,
        batch_size: int = 64,
    ) -> list[list[dict]]:
        """Vectorised retrieval for many queries (used by the evaluator).

        Encodes all ``samples`` at once, then issues a single batched index
        search, applying per-query leave-one-out exclusions.  Returns, per query,
        the same list-of-dict structure as :meth:`query` (so the evaluator can
        read off retrieved labels for F1@K).

        Parameters
        ----------
        samples:
            ``Sample`` list (queries).  Raw embeddings are also accepted as a
            ``(N, D)`` array, in which case no exclusions are applied.
        k:
            Results per query.
        gallery_modality, exclude_same_location, rerank:
            As in :meth:`query`.
        batch_size:
            Backbone batch size for encoding.

        Returns
        -------
        list[list[dict]]
            Outer length == number of queries.
        """
        if self.index is None:
            raise RuntimeError("batch_query() before index_gallery()")

        sample_inputs = (
            _is_sequence(samples) and len(samples) > 0 and _is_sample(samples[0])
        )
        if sample_inputs:
            q = self.encode(samples, batch_size=batch_size)
            excl = [
                self._exclusion_rows(s, exclude_same_location) for s in samples
            ]
            excl = [
                (np.empty(0, dtype=np.int64) if e is None else e) for e in excl
            ]
        else:
            q = _l2_normalize(np.asarray(samples, dtype=np.float32))
            excl = None

        strategy = self._resolve_rerank(rerank)
        if strategy is None:
            scores, idx = self.index.search(
                q, k, gallery_modality=gallery_modality, exclude_ids=excl
            )
        else:
            scores, idx = self.index.search_with_rerank(
                q,
                k,
                rerank=strategy,
                gallery_modality=gallery_modality,
                exclude_ids=excl,
            )
        return [self._format_hits(scores[i], idx[i]) for i in range(q.shape[0])]

    # ------------------------------------------------------------------
    # Evaluator-facing accessors
    # ------------------------------------------------------------------
    @property
    def gallery_embeddings(self) -> Optional[np.ndarray]:
        """The indexed gallery embedding matrix ``(N, D)`` (or ``None``)."""
        return None if self.index is None else self.index.embeddings

    def get_index(self) -> Optional[RetrievalIndex]:
        """Return the underlying :class:`RetrievalIndex`."""
        return self.index

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n = 0 if self.index is None else len(self.index)
        return (
            f"RetrievalEngine(gallery={n}, dim={self.embed_dim}, "
            f"whitener={'yes' if self.whitener else 'no'}, "
            f"rerank={self.rerank_default!r})"
        )


def _is_sequence(obj: Any) -> bool:
    """True for list/tuple inputs (but not numpy arrays or strings)."""
    return isinstance(obj, (list, tuple))
