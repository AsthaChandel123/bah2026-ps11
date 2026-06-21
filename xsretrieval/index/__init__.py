"""Fast retrieval indexes and re-ranking (faiss optional, numpy fallback).

Public API
----------
* :class:`RetrievalIndex`          — shared cross-modal vector index with a faiss
  backend and an **exact numpy brute-force fallback** (works without faiss).
  Supports modality-filtered search, leave-one-out exclusion, and a
  search→exact-refine→k-reciprocal re-rank path.
* :class:`ITQHasher`               — ITQ binary hashing of float embeddings.
* :class:`BinaryIndex`             — Hamming-distance index (numpy popcount or
  faiss binary MultiHash) — the near-O(1) coarse filter.
* :func:`hamming_topk`, :func:`pack_bits`, :func:`unpack_bits` — hashing helpers.
* Re-ranking (numpy): :func:`k_reciprocal_rerank`, :func:`exact_refine`,
  :func:`average_query_expansion`, :func:`alpha_qe`.

faiss is imported lazily *inside* methods; every numpy path is fully functional
on its own and returns exact results.
"""

from __future__ import annotations

from xsretrieval.index.faiss_index import RetrievalIndex
from xsretrieval.index.hashing import (
    BinaryIndex,
    ITQHasher,
    hamming_topk,
    pack_bits,
    unpack_bits,
)
from xsretrieval.index.rerank import (
    alpha_qe,
    average_query_expansion,
    exact_refine,
    k_reciprocal_rerank,
)

__all__ = [
    "RetrievalIndex",
    "ITQHasher",
    "BinaryIndex",
    "hamming_topk",
    "pack_bits",
    "unpack_bits",
    "k_reciprocal_rerank",
    "exact_refine",
    "average_query_expansion",
    "alpha_qe",
]
