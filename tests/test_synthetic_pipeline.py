"""End-to-end synthetic pipeline test for the evaluation benchmark.

Builds a small *solvable* synthetic multimodal dataset, wires it through a
:class:`RetrievalEngine` backed by the dependency-light ``FallbackBackbone``,
indexes the gallery, runs :func:`xsretrieval.eval.benchmark.evaluate`, and
asserts the four PS-11 headline F1 numbers are valid, *clearly above random
chance*, produced for **both** same- and cross-modal directions, and that
latency is positive.

This exercises the **eval** code (under test) against the real retrieval engine.
It depends on sibling-team modules (``xsretrieval.data.synthetic``,
``xsretrieval.models.backbones.fallback``, ``xsretrieval.retrieval``), so it
skips gracefully until those land. It is kept tiny (small synthetic set) so it
runs in well under a second, and the FallbackBackbone is pure numpy so torch is
never required.
"""

from __future__ import annotations

import numpy as np
import pytest

# Sibling-team dependencies (skip cleanly if absent). Eval itself needs numpy.
synthetic = pytest.importorskip(
    "xsretrieval.data.synthetic",
    reason="data team's synthetic generator not available yet",
)
fallback_mod = pytest.importorskip(
    "xsretrieval.models.backbones.fallback",
    reason="models team's FallbackBackbone not available yet",
)
engine_mod = pytest.importorskip(
    "xsretrieval.retrieval.engine",
    reason="retrieval team's RetrievalEngine not available yet",
)

from xsretrieval.data.modalities import Modality  # noqa: E402


def _make_query_gallery():
    """Build a solvable synthetic split: gallery = all, queries = a held-in subset.

    ``make_synthetic_multimodal`` returns a flat list of ``Sample`` (one per
    (class, location, modality)) sharing a ``location_id`` across modalities. We
    use the **entire** set as the gallery and pick a few queries per (class,
    modality) from within it -- the standard retrieval protocol where each query
    is excluded from its own results via leave-one-out (the engine drops the
    query's own id and co-located items, and the benchmark mirrors that in R_q).
    """
    make = synthetic.make_synthetic_multimodal
    # Small but enough relevants per class so F1@5/@10 are meaningful.
    samples = make(
        n_classes=4,
        per_class_per_modality=8,
        modalities=(Modality.OPTICAL_RGB, Modality.MULTISPECTRAL, Modality.SAR),
        size=32,
        seed=0,
    )
    assert len(samples) == 4 * 8 * 3

    gallery = list(samples)

    # Queries: the first 2 locations of each class, all modalities (held-in).
    queries = [
        s
        for s in samples
        if s.location_id.endswith("loc0") or s.location_id.endswith("loc1")
    ]
    assert queries and gallery
    return queries, gallery


def test_synthetic_end_to_end_above_chance() -> None:
    """The four headline F1s must be valid and clearly above random chance."""
    from xsretrieval.eval.benchmark import evaluate, format_report

    backbone = fallback_mod.FallbackBackbone()
    engine = engine_mod.RetrievalEngine(backbone)
    queries, gallery = _make_query_gallery()

    results = evaluate(
        engine,
        queries,
        gallery,
        ks=(5, 10),
        recall_mode="raw",
        measure_latency=True,
        latency_warmup=10,
        latency_runs=50,
    )

    headline = results["headline"]
    # All four headline F1s present and in [0, 1].
    for key in ("F1@5_same", "F1@10_same", "F1@5_cross", "F1@10_cross"):
        assert key in headline, f"missing headline metric {key}"
        assert 0.0 <= headline[key] <= 1.0, (
            f"{key}={headline[key]} out of [0,1]\n{format_report(results)}"
        )

    # Same- AND cross-modal numbers must both be produced.
    assert results["same_modal"]["n_cells"] > 0
    assert results["cross_modal"]["n_cells"] > 0

    # Clearly above random chance for same-modal (the easy direction). With 4
    # classes random F1@5 would be far below 0.15.
    assert headline["F1@5_same"] > 0.15, (
        f"F1@5_same={headline['F1@5_same']:.4f} not above chance\n"
        f"{format_report(results)}"
    )
    # The fallback backbone preserves cross-modal structure too; require the
    # cross-modal headline to be a valid, positive, non-trivial number.
    assert headline["F1@5_cross"] > 0.0, (
        f"F1@5_cross collapsed to 0\n{format_report(results)}"
    )

    # Latency must be measured and positive, with sane percentile ordering.
    assert results["avg_query_time_ms"] is not None
    assert results["avg_query_time_ms"] > 0.0
    lat = results["latency"]
    assert lat["p95_query_time_ms"] >= lat["p50_query_time_ms"]
    assert lat["n_runs"] >= 1

    # The report should render and contain the headline + a recall<=1 sanity.
    report = format_report(results)
    assert "headline" in report.lower()
    assert "F1@5" in report
    # Recall can never exceed 1 in any cell (LOO/R_q consistency check).
    for cell in results["cells"].values():
        assert cell["R@5"] <= 1.0 + 1e-9, f"recall>1 in cell {cell}"
        assert cell["R@10"] <= 1.0 + 1e-9, f"recall>1 in cell {cell}"


def test_synthetic_evaluate_capped_mode_and_no_latency() -> None:
    """`recall_mode='capped'` and `measure_latency=False` both work."""
    from xsretrieval.eval.benchmark import evaluate

    backbone = fallback_mod.FallbackBackbone()
    engine = engine_mod.RetrievalEngine(backbone)
    queries, gallery = _make_query_gallery()

    results = evaluate(
        engine,
        queries,
        gallery,
        ks=(5, 10),
        recall_mode="capped",
        measure_latency=False,
    )
    assert results["avg_query_time_ms"] is None
    for key in ("F1@5_same", "F1@10_same", "F1@5_cross", "F1@10_cross"):
        assert 0.0 <= results["headline"][key] <= 1.0
