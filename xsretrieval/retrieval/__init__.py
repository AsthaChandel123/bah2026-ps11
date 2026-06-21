"""High-level retrieval engine (encode → whiten → index → query).

Public API
----------
* :class:`RetrievalEngine` — duck-typed, CPU-fast engine that wires a backbone,
  optional projection head and optional whitener to a :class:`RetrievalIndex`,
  exposing ``encode`` / ``index_gallery`` / ``query`` / ``batch_query`` /
  ``fit_whitener`` for inference and evaluation.

Imports no torch/faiss at module level; the numpy brute-force index path gives
exact retrieval results without either dependency installed.
"""

from __future__ import annotations

from xsretrieval.retrieval.engine import RetrievalEngine

__all__ = ["RetrievalEngine"]
