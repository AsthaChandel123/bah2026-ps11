"""End-to-end retrieval benchmark for cross-modal satellite image retrieval.

This module turns a :class:`RetrievalEngine` (duck-typed; owned by the retrieval
team) plus query/gallery :class:`~xsretrieval.data.modalities.Sample` lists into
the **four headline F1 numbers** that PS-11 is scored on -- F1@5/F1@10 for
same-modal and cross-modal retrieval -- together with the full
query-modality x gallery-modality evaluation matrix, mAP per cell, and a
correctly-measured average query latency.

Engine contract (duck-typed -- only these two methods are used)
---------------------------------------------------------------
``engine.index_gallery(samples) -> None``
    Embed and index a list of ``Sample`` objects. May be called more than once;
    the last call's gallery is what subsequent queries search.

``engine.batch_query(samples, k, gallery_modality=None, rerank=False) -> \
list[list[dict]]``
    For each query ``Sample`` return a ranked list (best first) of result dicts
    with keys ``{"id", "label", "modality", "score"}``. ``gallery_modality``
    restricts results to a single target modality (the cross-modal evaluation
    cell); ``None`` means "all modalities". ``rerank`` toggles the optional
    re-ranking stage. We always over-fetch ``k + buffer`` so that leave-one-out
    removal of the query's own item/location still leaves ``k`` results.

Everything here is plain Python + numpy; heavy libraries are never imported.
The metric maths is delegated to :mod:`xsretrieval.eval.metrics` (pure numpy,
independently unit-tested).
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any, Iterable, Optional, Protocol, Sequence

import numpy as np

from xsretrieval.data.modalities import Modality

from .metrics import mean_metrics

__all__ = [
    "RetrievalEngineLike",
    "build_relevance",
    "evaluate",
    "format_report",
    "latency_benchmark",
]

# Over-fetch this many extra results beyond k so leave-one-out exclusion of the
# query's own id/location cannot starve the top-k.
_LOO_BUFFER = 8

# The two cutoffs PS-11 scores; the headline numbers are F1@5 and F1@10.
DEFAULT_KS: tuple[int, ...] = (5, 10)


class RetrievalEngineLike(Protocol):
    """Structural type for the retrieval engine (see module docstring)."""

    def index_gallery(self, samples: Sequence[Any]) -> Any:  # pragma: no cover
        ...

    def batch_query(  # pragma: no cover
        self,
        samples: Sequence[Any],
        k: int,
        gallery_modality: Optional[Modality] = None,
        **kwargs: Any,
    ) -> "list[list[dict]]":
        ...


# ---------------------------------------------------------------------------
# Relevance construction
# ---------------------------------------------------------------------------
def build_relevance(gallery_labels: Iterable[int]) -> dict[int, int]:
    """Map each class label to its member count ``R_q`` in the gallery.

    Under class-equality relevance, the total number of relevant items for a
    query of class ``c`` is simply the number of gallery items with label ``c``.
    This returns that lookup table; callers subtract 1 for leave-one-out when
    the query itself lives in the gallery (see :func:`evaluate`).

    Labels ``< 0`` (the "unknown/unlabeled" sentinel of
    :class:`~xsretrieval.data.modalities.Sample`) are ignored.

    Parameters
    ----------
    gallery_labels:
        Iterable of integer class labels of the gallery items.

    Returns
    -------
    dict mapping ``label -> count``.
    """
    counts: Counter[int] = Counter(
        int(lbl) for lbl in gallery_labels if int(lbl) >= 0
    )
    return dict(counts)


# ---------------------------------------------------------------------------
# Internal: turn ranked result dicts into a padded label matrix + R_q vector
# ---------------------------------------------------------------------------
def _modality_of(sample: Any) -> Modality:
    """Return the ``Modality`` of a sample, coercing raw strings if needed."""
    m = sample.modality
    return m if isinstance(m, Modality) else Modality(m)


def _filter_leave_one_out(
    results: "list[dict]", query: Any, k: int
) -> "list[dict]":
    """Drop the query's own item and co-located items, then keep the top-k.

    "Leave-one-out" for retrieval means a query image is never relevant to
    itself: we remove any result whose ``id`` equals the query's id, and -- when
    the query carries a ``location_id`` -- any result sharing that location (the
    same geographic tile observed by the same/other sensor), since a co-located
    item is the query's own ground-truth pair, not a retrieved neighbour to be
    scored. The remaining ranked list is truncated to ``k``.
    """
    q_id = query.id
    q_loc = getattr(query, "location_id", None)
    kept: list[dict] = []
    for r in results:
        if r.get("id") == q_id:
            continue
        if q_loc is not None and r.get("location_id", None) == q_loc:
            # Only present if the engine echoes location_id; harmless otherwise.
            continue
        kept.append(r)
        if len(kept) >= k:
            break
    return kept


def _labels_matrix(
    per_query_results: "list[list[dict]]",
    queries: Sequence[Any],
    k: int,
) -> np.ndarray:
    """Build a ``(Q, k)`` int label matrix from ranked result dicts.

    Each row holds the labels of the top-``k`` retrieved items (after
    leave-one-out). Rows shorter than ``k`` are padded with a sentinel label
    ``-1`` that can never equal a real (``>= 0``) query label, so the padded
    slots correctly count as non-relevant in the metrics.
    """
    q = len(queries)
    out = np.full((q, k), -1, dtype=np.int64)
    for i, (query, results) in enumerate(zip(queries, per_query_results)):
        kept = _filter_leave_one_out(results, query, k)
        for j, r in enumerate(kept[:k]):
            out[i, j] = int(r.get("label", -1))
    return out


def _relevant_counts_for_queries(
    queries: Sequence[Any],
    gallery_label_counts: dict[int, int],
    gallery_ids: Optional[set] = None,
    gallery_locations: Optional[set] = None,
) -> np.ndarray:
    """Compute ``R_q`` per query under class-equality relevance with LOO.

    ``R_q`` for a query of class ``c`` is the number of gallery items of class
    ``c`` *in the target modality cell*, minus the query's own contribution when
    it is genuinely present in that cell (leave-one-out). Presence is detected
    by **actual membership** -- the query's ``id`` (or, failing that, its
    ``location_id``) appears in the cell's id/location sets -- rather than
    assumed from modality equality. This keeps ``R_q`` exactly consistent with
    what the engine can return after the same leave-one-out filtering, so recall
    can never exceed 1 whether queries are held out from, or drawn from, the
    gallery.

    Parameters
    ----------
    queries:
        The query samples.
    gallery_label_counts:
        ``label -> count`` for the gallery **restricted to the target
        modality** cell.
    gallery_ids:
        Set of item ids present in the target modality cell (for LOO by id).
    gallery_locations:
        Set of ``location_id`` values present in the cell (for LOO by location,
        used only when the query id itself is not in the gallery).

    Returns
    -------
    ``(Q,)`` int array of ``R_q`` values (clipped at 0).
    """
    gallery_ids = gallery_ids or set()
    gallery_locations = gallery_locations or set()
    q = len(queries)
    r_q = np.zeros(q, dtype=np.int64)
    for i, query in enumerate(queries):
        lbl = int(query.label)
        count = gallery_label_counts.get(lbl, 0)
        # Leave-one-out: subtract the query's own item only if it is actually a
        # member of this gallery cell (same id, or same geographic location).
        q_loc = getattr(query, "location_id", None)
        if query.id in gallery_ids:
            count -= 1
        elif q_loc is not None and q_loc in gallery_locations:
            count -= 1
        r_q[i] = max(count, 0)
    return r_q


# ---------------------------------------------------------------------------
# Core: per-cell evaluation
# ---------------------------------------------------------------------------
def _gallery_index_by_modality(
    gallery_samples: Sequence[Any],
) -> dict[Modality, dict[str, Any]]:
    """Per-modality gallery index: class counts + id/location membership sets.

    Returns ``{modality: {"counts": {label: n}, "ids": {...}, "locations":
    {...}}}`` for every modality present in the gallery. The id/location sets
    power exact leave-one-out (see :func:`_relevant_counts_for_queries`).
    """
    by_mod_labels: dict[Modality, list[int]] = {}
    by_mod_ids: dict[Modality, set] = {}
    by_mod_locs: dict[Modality, set] = {}
    for s in gallery_samples:
        m = _modality_of(s)
        by_mod_labels.setdefault(m, []).append(int(s.label))
        by_mod_ids.setdefault(m, set()).add(s.id)
        loc = getattr(s, "location_id", None)
        if loc is not None:
            by_mod_locs.setdefault(m, set()).add(loc)
    out: dict[Modality, dict[str, Any]] = {}
    for m, labels in by_mod_labels.items():
        out[m] = {
            "counts": build_relevance(labels),
            "ids": by_mod_ids.get(m, set()),
            "locations": by_mod_locs.get(m, set()),
        }
    return out


def _queries_by_modality(
    query_samples: Sequence[Any],
) -> dict[Modality, list[Any]]:
    """Group query samples by their modality."""
    groups: dict[Modality, list[Any]] = {}
    for s in query_samples:
        groups.setdefault(_modality_of(s), []).append(s)
    return groups


def _evaluate_cell(
    engine: RetrievalEngineLike,
    cell_queries: Sequence[Any],
    gallery_modality: Modality,
    gallery_cell_index: dict[str, Any],
    ks: Sequence[int],
    recall_mode: str,
    rerank: bool,
) -> dict[str, Any]:
    """Evaluate one (query_modality, gallery_modality) cell.

    Runs a single batched query restricted to ``gallery_modality``, over-fetches
    ``max(ks) + buffer`` so leave-one-out cannot starve the top-k, then computes
    macro-averaged metrics at every cutoff in ``ks``.
    """
    max_k = max(ks)
    fetch_k = max_k + _LOO_BUFFER
    results = engine.batch_query(
        cell_queries,
        fetch_k,
        gallery_modality=gallery_modality,
        rerank=rerank,
    )
    r_q = _relevant_counts_for_queries(
        cell_queries,
        gallery_cell_index.get("counts", {}),
        gallery_ids=gallery_cell_index.get("ids", set()),
        gallery_locations=gallery_cell_index.get("locations", set()),
    )
    q_labels = np.array([int(s.label) for s in cell_queries], dtype=np.int64)

    cell: dict[str, Any] = {"n_queries": len(cell_queries)}
    max_k = max(ks)
    for k in ks:
        labels_2d = _labels_matrix(results, cell_queries, k)
        m = mean_metrics(labels_2d, q_labels, r_q, k=k, mode=recall_mode)
        cell[f"P@{k}"] = m[f"P@{k}"]
        cell[f"R@{k}"] = m[f"R@{k}"]
        cell[f"F1@{k}"] = m[f"F1@{k}"]
        cell[f"nDCG@{k}"] = m[f"nDCG@{k}"]
        # mAP / MRR summarise the *deepest* retrieved list (max k): compute them
        # once from that pass so they are well-defined regardless of ks ordering.
        if k == max_k:
            cell["mAP"] = m["mAP"]
            cell["MRR"] = m["MRR"]
    return cell


def _aggregate(
    cells: dict[tuple[Modality, Modality], dict[str, Any]],
    ks: Sequence[int],
    same_modal: bool,
) -> dict[str, float]:
    """Macro-average a set of cells (diagonal or off-diagonal) per metric.

    Cells are weighted equally (macro over cells), matching how PS-11 averages
    same-modal (diagonal) and cross-modal (off-diagonal) scores. Empty cells (no
    queries) are skipped so they cannot drag the average toward zero.
    """
    selected = [
        c
        for (qm, gm), c in cells.items()
        if (qm == gm) == same_modal and c.get("n_queries", 0) > 0
    ]
    agg: dict[str, float] = {}
    metric_keys: list[str] = []
    for k in ks:
        metric_keys += [f"P@{k}", f"R@{k}", f"F1@{k}", f"nDCG@{k}"]
    metric_keys += ["mAP", "MRR"]
    for key in metric_keys:
        vals = [c[key] for c in selected if key in c]
        agg[key] = float(np.mean(vals)) if vals else 0.0
    agg["n_cells"] = float(len(selected))
    return agg


# ---------------------------------------------------------------------------
# Latency measurement (done correctly: warmup, batch=1, search-only, p50/p95)
# ---------------------------------------------------------------------------
def latency_benchmark(
    index_or_engine: Any,
    queries: Sequence[Any],
    n_warmup: int = 50,
    n_runs: int = 500,
    *,
    k: int = 10,
    gallery_modality: Optional[Modality] = None,
    rerank: bool = False,
) -> dict[str, float]:
    """Measure single-query (batch=1) search latency, the PS-11 5th metric.

    Methodology (``research/04`` Part 15):

    1. **Warm up** with at least ``n_warmup`` throwaway single queries (page-in
       caches, JIT, thread pools) -- their times are discarded.
    2. **Time batch=1** ``search`` only (the engine's ``batch_query`` with a
       single sample). Index build/add is excluded.
    3. Repeat ``n_runs`` times, cycling through ``queries`` if necessary, and
       report **mean, p50 and p95** in milliseconds.

    When ``rerank=True`` the re-ranking stage runs *inside* the timed window
    (the metric must include re-ranking), because it is part of the query path.

    Parameters
    ----------
    index_or_engine:
        An object exposing ``batch_query`` (the retrieval engine). A raw index
        with the same method also works (duck-typed).
    queries:
        Query samples to time against (cycled if fewer than ``n_runs``).
    n_warmup, n_runs:
        Warmup and timed iteration counts.
    k:
        Top-k to retrieve while timing.
    gallery_modality:
        Optional target-modality filter (kept consistent with the eval cell).
    rerank:
        Include the re-ranking stage in the timed path.

    Returns
    -------
    dict with ``avg_query_time_ms``, ``p50_query_time_ms``,
    ``p95_query_time_ms``, ``min_query_time_ms``, ``max_query_time_ms`` and the
    settings ``n_warmup`` / ``n_runs`` (as floats) actually used.
    """
    if not queries:
        return {
            "avg_query_time_ms": 0.0,
            "p50_query_time_ms": 0.0,
            "p95_query_time_ms": 0.0,
            "min_query_time_ms": 0.0,
            "max_query_time_ms": 0.0,
            "n_warmup": 0.0,
            "n_runs": 0.0,
        }

    query_fn = index_or_engine.batch_query
    n_q = len(queries)

    def _one(idx: int) -> None:
        sample = queries[idx % n_q]
        query_fn(
            [sample], k, gallery_modality=gallery_modality, rerank=rerank
        )

    # --- warmup (discard) ---
    for i in range(max(0, int(n_warmup))):
        _one(i)

    # --- timed, batch=1 ---
    runs = max(1, int(n_runs))
    timings = np.empty(runs, dtype=np.float64)
    for i in range(runs):
        t0 = time.perf_counter()
        _one(i)
        timings[i] = (time.perf_counter() - t0) * 1e3  # ms

    return {
        "avg_query_time_ms": float(np.mean(timings)),
        "p50_query_time_ms": float(np.percentile(timings, 50)),
        "p95_query_time_ms": float(np.percentile(timings, 95)),
        "min_query_time_ms": float(np.min(timings)),
        "max_query_time_ms": float(np.max(timings)),
        "n_warmup": float(max(0, int(n_warmup))),
        "n_runs": float(runs),
    }


# ---------------------------------------------------------------------------
# Top-level evaluation
# ---------------------------------------------------------------------------
def evaluate(
    engine: RetrievalEngineLike,
    query_samples: Sequence[Any],
    gallery_samples: Sequence[Any],
    ks: Sequence[int] = DEFAULT_KS,
    recall_mode: str = "raw",
    measure_latency: bool = True,
    rerank: bool = False,
    *,
    latency_warmup: int = 50,
    latency_runs: int = 500,
) -> dict[str, Any]:
    """Run the full PS-11 evaluation and return a structured result dict.

    Steps
    -----
    1. **Index the gallery** once (``engine.index_gallery``).
    2. **Build the evaluation matrix**: for every (query modality, gallery
       modality) pair present in the data, restrict the gallery to that target
       modality, run ``engine.batch_query`` with the ``gallery_modality``
       filter, and compute macro-averaged P/R/F1/nDCG@k + mAP for the cell.
       The query's own item/location is excluded (leave-one-out).
    3. **Aggregate** the diagonal cells (``q_mod == g_mod``) into the same-modal
       headline numbers and the off-diagonal cells (``q_mod != g_mod``) into the
       cross-modal headline numbers -> ``F1@5_same``, ``F1@10_same``,
       ``F1@5_cross``, ``F1@10_cross``.
    4. **Measure latency** correctly (warmup, batch=1, search-only, p50/p95) if
       ``measure_latency``; the timer includes re-ranking when ``rerank=True``.

    Parameters
    ----------
    engine:
        Retrieval engine (duck-typed; see module docstring).
    query_samples, gallery_samples:
        Lists of :class:`~xsretrieval.data.modalities.Sample`.
    ks:
        Cutoffs to evaluate. PS-11 scores ``(5, 10)``; the four headline F1s are
        always taken from ``k=5`` and ``k=10`` if present.
    recall_mode:
        ``"raw"`` (reported) or ``"capped"`` (fair model-selection); see
        :func:`~xsretrieval.eval.metrics.recall_at_k`.
    measure_latency:
        Whether to run the standalone latency benchmark.
    rerank:
        Pass-through to the engine and into the latency timer.
    latency_warmup, latency_runs:
        Forwarded to :func:`latency_benchmark`.

    Returns
    -------
    dict with keys:

    * ``"cells"`` -- ``{"<qmod>__<gmod>": {metrics...}}`` per matrix cell.
    * ``"same_modal"`` / ``"cross_modal"`` -- aggregated metric dicts.
    * ``"headline"`` -- ``{"F1@5_same", "F1@10_same", "F1@5_cross",
      "F1@10_cross"}`` (cutoffs that are missing from ``ks`` are omitted).
    * ``"latency"`` -- the :func:`latency_benchmark` dict (if measured).
    * ``"avg_query_time_ms"`` -- convenience copy of the mean latency.
    * ``"config"`` -- echo of ``ks`` / ``recall_mode`` / ``rerank`` and the
      modality lists discovered.
    """
    ks = tuple(int(k) for k in ks)
    if not ks:
        raise ValueError("ks must contain at least one cutoff")

    # 1) Index the gallery once.
    engine.index_gallery(list(gallery_samples))

    # Pre-compute per-modality gallery index (counts + LOO sets) and query
    # groups.
    gallery_index_by_mod = _gallery_index_by_modality(gallery_samples)
    gallery_modalities = sorted(
        gallery_index_by_mod.keys(), key=lambda m: m.value
    )
    query_groups = _queries_by_modality(query_samples)
    query_modalities = sorted(query_groups.keys(), key=lambda m: m.value)

    # 2) Build the evaluation matrix cell by cell.
    cells: dict[tuple[Modality, Modality], dict[str, Any]] = {}
    for q_mod in query_modalities:
        cell_queries = query_groups[q_mod]
        for g_mod in gallery_modalities:
            cell_index = gallery_index_by_mod.get(g_mod, {})
            cell = _evaluate_cell(
                engine,
                cell_queries,
                g_mod,
                cell_index,
                ks,
                recall_mode,
                rerank,
            )
            cell["query_modality"] = q_mod.value
            cell["gallery_modality"] = g_mod.value
            cell["same_modal"] = q_mod == g_mod
            cells[(q_mod, g_mod)] = cell

    # 3) Aggregate diagonal (same) and off-diagonal (cross) cells.
    same_agg = _aggregate(cells, ks, same_modal=True)
    cross_agg = _aggregate(cells, ks, same_modal=False)

    headline: dict[str, float] = {}
    for k in ks:
        if k in (5, 10):
            headline[f"F1@{k}_same"] = same_agg.get(f"F1@{k}", 0.0)
            headline[f"F1@{k}_cross"] = cross_agg.get(f"F1@{k}", 0.0)

    # Serialise cells with string keys for a JSON-friendly result.
    cells_out: dict[str, dict[str, Any]] = {
        f"{qm.value}__{gm.value}": cell for (qm, gm), cell in cells.items()
    }

    result: dict[str, Any] = {
        "cells": cells_out,
        "same_modal": same_agg,
        "cross_modal": cross_agg,
        "headline": headline,
        "config": {
            "ks": list(ks),
            "recall_mode": recall_mode,
            "rerank": bool(rerank),
            "query_modalities": [m.value for m in query_modalities],
            "gallery_modalities": [m.value for m in gallery_modalities],
            "n_query_samples": len(query_samples),
            "n_gallery_samples": len(gallery_samples),
        },
    }

    # 4) Latency (correctly measured).
    if measure_latency:
        # Time against a representative slice of queries on the full mixed
        # gallery (gallery_modality=None) -- the native serving path.
        timing_queries = list(query_samples)
        lat = latency_benchmark(
            engine,
            timing_queries,
            n_warmup=latency_warmup,
            n_runs=latency_runs,
            k=max(ks),
            gallery_modality=None,
            rerank=rerank,
        )
        result["latency"] = lat
        result["avg_query_time_ms"] = lat["avg_query_time_ms"]
    else:
        result["avg_query_time_ms"] = None

    return result


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------
def _fmt(x: Optional[float], width: int = 7, prec: int = 4) -> str:
    """Format a float (or ``None``/NaN) for the report table."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a".rjust(width)
    return f"{x:>{width}.{prec}f}"


def format_report(results: dict[str, Any]) -> str:
    """Render a clean text report of an :func:`evaluate` result.

    Produces (in order): a header, the **evaluation matrix** as a table of
    F1@k per (query, gallery) modality cell, the **same/cross headline F1**
    block, and the **latency** summary. The output is plain ASCII suitable for
    printing in the CLI and pasting into the README.

    Parameters
    ----------
    results:
        The dict returned by :func:`evaluate`.

    Returns
    -------
    str -- the formatted multi-line report.
    """
    cfg = results.get("config", {})
    ks = cfg.get("ks", list(DEFAULT_KS))
    cells: dict[str, dict[str, Any]] = results.get("cells", {})
    lines: list[str] = []

    title = "Cross-Modal Satellite Image Retrieval - Evaluation Report"
    lines.append("=" * len(title))
    lines.append(title)
    lines.append("=" * len(title))
    lines.append(
        f"queries={cfg.get('n_query_samples', '?')}  "
        f"gallery={cfg.get('n_gallery_samples', '?')}  "
        f"recall_mode={cfg.get('recall_mode', '?')}  "
        f"rerank={cfg.get('rerank', '?')}"
    )
    lines.append("")

    # --- Evaluation matrix (one row per cell) ---
    lines.append("Evaluation matrix (query x gallery modality)")
    lines.append("-" * 78)
    metric_cols = []
    for k in ks:
        metric_cols += [f"P@{k}", f"R@{k}", f"F1@{k}"]
    metric_cols += ["mAP"]
    header = f"{'query':<14}{'gallery':<14}{'type':<7}"
    header += "".join(f"{c:>9}" for c in metric_cols)
    lines.append(header)
    lines.append("-" * 78)

    # Stable ordering: same-modal cells first, then cross-modal.
    def _cell_sort_key(item: tuple[str, dict[str, Any]]):
        _, c = item
        return (not c.get("same_modal", False), c.get("query_modality", ""),
                c.get("gallery_modality", ""))

    for _key, c in sorted(cells.items(), key=_cell_sort_key):
        qm = str(c.get("query_modality", "?"))
        gm = str(c.get("gallery_modality", "?"))
        typ = "same" if c.get("same_modal") else "cross"
        row = f"{qm:<14}{gm:<14}{typ:<7}"
        for col in metric_cols:
            row += _fmt(c.get(col), width=9)
        lines.append(row)
    lines.append("-" * 78)
    lines.append("")

    # --- Aggregated headline metrics ---
    same = results.get("same_modal", {})
    cross = results.get("cross_modal", {})
    lines.append("Aggregated headline metrics")
    lines.append("-" * 78)
    lines.append(
        f"{'aggregate':<14}"
        + "".join(f"{c:>9}" for c in metric_cols)
        + f"{'n_cells':>9}"
    )
    for name, agg in (("same-modal", same), ("cross-modal", cross)):
        row = f"{name:<14}"
        for col in metric_cols:
            row += _fmt(agg.get(col), width=9)
        row += f"{int(agg.get('n_cells', 0)):>9d}"
        lines.append(row)
    lines.append("-" * 78)
    lines.append("")

    # --- Four headline F1 numbers ---
    headline = results.get("headline", {})
    lines.append("PS-11 headline F1 scores")
    lines.append("-" * 40)
    for k in ks:
        if k in (5, 10):
            same_v = headline.get(f"F1@{k}_same")
            cross_v = headline.get(f"F1@{k}_cross")
            lines.append(
                f"  F1@{k:<2}  same-modal : {_fmt(same_v)}    "
                f"cross-modal : {_fmt(cross_v)}"
            )
    lines.append("-" * 40)
    lines.append("")

    # --- Latency ---
    lat = results.get("latency")
    lines.append("Average retrieval time per query (batch=1, search-only)")
    lines.append("-" * 78)
    if lat:
        lines.append(
            f"  mean : {lat['avg_query_time_ms']:.3f} ms"
            f"   p50 : {lat['p50_query_time_ms']:.3f} ms"
            f"   p95 : {lat['p95_query_time_ms']:.3f} ms"
        )
        lines.append(
            f"  warmup={int(lat.get('n_warmup', 0))} "
            f"runs={int(lat.get('n_runs', 0))} "
            f"(rerank in timer: {cfg.get('rerank', False)})"
        )
    else:
        lines.append("  (latency not measured)")
    lines.append("-" * 78)

    return "\n".join(lines)
