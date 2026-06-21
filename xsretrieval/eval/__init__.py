"""Evaluation subpackage: scored metrics + end-to-end retrieval benchmark.

This package provides the **scored core** of BAH 2026 PS-11:

* :mod:`xsretrieval.eval.metrics` -- pure-numpy retrieval metrics
  (Precision@K, Recall@K raw/capped, F1@K with closed-form cross-check,
  Average Precision, nDCG@K, Reciprocal Rank, and vectorised batch helpers).
  No heavy dependencies, so it imports and unit-tests with only ``numpy``.
* :mod:`xsretrieval.eval.benchmark` -- builds the query x gallery evaluation
  matrix from a (duck-typed) retrieval engine, aggregates same-modal and
  cross-modal F1@5 / F1@10, measures average query latency correctly, and
  formats a human-readable report.

The metric maths is intentionally separated from the engine plumbing so the
former can be trusted in isolation (its correctness *is* the competition score).
"""

from __future__ import annotations

from .benchmark import (
    RetrievalEngineLike,
    build_relevance,
    evaluate,
    format_report,
    latency_benchmark,
)
from .metrics import (
    RECALL_MODES,
    average_precision,
    batch_f1_at_k,
    f1_at_k,
    mean_metrics,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance_from_labels,
)

__all__ = [
    # metrics
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
    # benchmark
    "RetrievalEngineLike",
    "build_relevance",
    "evaluate",
    "format_report",
    "latency_benchmark",
]
