"""Rigorous unit tests for the pure-numpy retrieval metrics.

These tests are the safety net for the **scored core** of PS-11. Every expected
value is computed by hand (and cross-checked against the closed-form formula),
on small hand-crafted ranked-label arrays. The headline assertion is the
closed-form F1@K identity ``F1@K == 2*r_K / (K + R_q)`` for raw recall, plus the
exact worked example from ``research/06_sota_evaluation.md`` Part 2D.

This file depends only on ``numpy`` + ``pytest`` and MUST pass standalone.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xsretrieval.eval.metrics import (
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

# A reusable "airport" query: query_label = 1, gallery has R_q = 8 relevants.
# Ranked relevance pattern from research/06 Part 2D: [1,1,1,0,1,1,0,1,0,1].
# Encode as labels where 1 == relevant (airport), 0 == some other class.
WORKED_LABELS = np.array([1, 1, 1, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
WORKED_QUERY_LABEL = 1
WORKED_RQ = 8


# ---------------------------------------------------------------------------
# Precision@K
# ---------------------------------------------------------------------------
class TestPrecisionAtK:
    def test_perfect_precision(self) -> None:
        labels = np.array([5, 5, 5, 5, 5], dtype=np.int64)
        assert precision_at_k(labels, 5, k=5) == 1.0

    def test_zero_precision(self) -> None:
        labels = np.array([0, 0, 0, 0, 0], dtype=np.int64)
        assert precision_at_k(labels, 7, k=5) == 0.0

    def test_partial_precision_hand_computed(self) -> None:
        # top-5 of the worked example: [1,1,1,0,1] -> r_5 = 4 -> 4/5 = 0.8
        assert precision_at_k(WORKED_LABELS, WORKED_QUERY_LABEL, k=5) == 0.8
        # top-10: r_10 = 7 -> 7/10 = 0.7
        assert precision_at_k(WORKED_LABELS, WORKED_QUERY_LABEL, k=10) == 0.7

    def test_k_larger_than_list_counts_empty_slots(self) -> None:
        # Only 3 items retrieved, all relevant, but k=5 -> 3/5 = 0.6.
        labels = np.array([2, 2, 2], dtype=np.int64)
        assert precision_at_k(labels, 2, k=5) == pytest.approx(0.6)

    def test_relevant_mask_form(self) -> None:
        mask = [True, False, True, True, False]
        assert precision_at_k(k=5, relevant_mask=mask) == pytest.approx(0.6)

    def test_invalid_k_raises(self) -> None:
        with pytest.raises(ValueError):
            precision_at_k(WORKED_LABELS, 1, k=0)

    def test_both_forms_supplied_raises(self) -> None:
        with pytest.raises(ValueError):
            precision_at_k(WORKED_LABELS, 1, k=5, relevant_mask=[True])


# ---------------------------------------------------------------------------
# Recall@K (raw and capped)
# ---------------------------------------------------------------------------
class TestRecallAtK:
    def test_raw_recall_hand_computed(self) -> None:
        # r_5 = 4, R_q = 8 -> 4/8 = 0.5
        assert recall_at_k(
            WORKED_LABELS, WORKED_QUERY_LABEL, WORKED_RQ, k=5, mode="raw"
        ) == 0.5
        # r_10 = 7, R_q = 8 -> 7/8 = 0.875
        assert recall_at_k(
            WORKED_LABELS, WORKED_QUERY_LABEL, WORKED_RQ, k=10, mode="raw"
        ) == pytest.approx(0.875)

    def test_capped_recall_differs_for_large_class(self) -> None:
        # r_5 = 4, min(5, 8) = 5 -> 4/5 = 0.8 (capped) vs 0.5 (raw)
        assert recall_at_k(
            WORKED_LABELS, WORKED_QUERY_LABEL, WORKED_RQ, k=5, mode="capped"
        ) == 0.8

    def test_capped_equals_raw_when_rq_le_k(self) -> None:
        # R_q = 8, k = 10 -> min(10,8)=8 == R_q, so raw == capped == 7/8.
        raw = recall_at_k(
            WORKED_LABELS, WORKED_QUERY_LABEL, WORKED_RQ, k=10, mode="raw"
        )
        capped = recall_at_k(
            WORKED_LABELS, WORKED_QUERY_LABEL, WORKED_RQ, k=10, mode="capped"
        )
        assert raw == capped == pytest.approx(0.875)

    def test_large_class_cap_math(self) -> None:
        # Perfect top-5 but R_q = 100: raw recall = 5/100 = 0.05.
        labels = np.full(5, 3, dtype=np.int64)
        assert recall_at_k(labels, 3, 100, k=5, mode="raw") == pytest.approx(0.05)
        # capped: min(5,100)=5 -> 5/5 = 1.0
        assert recall_at_k(labels, 3, 100, k=5, mode="capped") == 1.0

    def test_zero_total_relevant_returns_zero(self) -> None:
        assert recall_at_k(WORKED_LABELS, 1, 0, k=5) == 0.0

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            recall_at_k(WORKED_LABELS, 1, 8, k=5, mode="bogus")


# ---------------------------------------------------------------------------
# F1@K -- closed-form cross-check is the centrepiece
# ---------------------------------------------------------------------------
class TestF1AtK:
    def test_worked_example_at_5(self) -> None:
        # research/06 Part 2D: F1@5 = 0.615 (raw), via 2*4/(5+8) = 8/13.
        val = f1_at_k(
            WORKED_LABELS, WORKED_QUERY_LABEL, WORKED_RQ, k=5, mode="raw"
        )
        assert val == pytest.approx(8.0 / 13.0)
        assert val == pytest.approx(0.6153846153846154)

    def test_worked_example_at_10(self) -> None:
        # F1@10 = 0.778 (raw), via 2*7/(10+8) = 14/18.
        val = f1_at_k(
            WORKED_LABELS, WORKED_QUERY_LABEL, WORKED_RQ, k=10, mode="raw"
        )
        assert val == pytest.approx(14.0 / 18.0)
        assert val == pytest.approx(0.7777777777777778)

    def test_cross_modal_paired_example(self) -> None:
        # research/06: optical->SAR, R_q=1, top-5 = [0,1,0,0,0] -> r_5=1.
        # F1@5 = 2*1/(5+1) = 1/3.
        labels = np.array([0, 1, 0, 0, 0], dtype=np.int64)
        val = f1_at_k(labels, 1, total_relevant=1, k=5, mode="raw")
        assert val == pytest.approx(1.0 / 3.0)

    def test_small_class_perfect_ranking(self) -> None:
        # R_q = 2, k = 5, perfect: top-5 has the 2 relevants first.
        # P@5 = 2/5, R@5 = 1.0 -> F1 = 2*2/(5+2) = 4/7 = 0.5714...
        labels = np.array([9, 9, 0, 0, 0], dtype=np.int64)
        val = f1_at_k(labels, 9, total_relevant=2, k=5, mode="raw")
        assert val == pytest.approx(4.0 / 7.0)

    @pytest.mark.parametrize(
        "labels,query,rq,k",
        [
            (WORKED_LABELS, 1, 8, 5),
            (WORKED_LABELS, 1, 8, 10),
            (np.array([0, 1, 0, 0, 0]), 1, 1, 5),
            (np.array([3, 3, 3, 3, 3, 3]), 3, 50, 5),
            (np.array([2, 0, 2, 0, 2, 0, 2]), 2, 4, 3),
            (np.array([1, 1]), 1, 2, 5),  # short list, k > len
        ],
    )
    def test_harmonic_mean_equals_closed_form_raw(
        self, labels, query, rq, k
    ) -> None:
        """F1 via harmonic mean must equal the closed form 2*r_K/(K+R_q)."""
        labels = np.asarray(labels)
        r_k = int(np.count_nonzero(labels[:k] == query))
        closed = 2.0 * r_k / (k + rq) if rq > 0 else 0.0
        via_hm = f1_at_k(labels, query, rq, k=k, mode="raw")
        assert via_hm == pytest.approx(closed), (
            f"harmonic-mean F1 {via_hm} != closed form {closed} "
            f"(r_K={r_k}, K={k}, R_q={rq})"
        )

    @pytest.mark.parametrize(
        "labels,query,rq,k",
        [
            (WORKED_LABELS, 1, 8, 5),
            (np.array([3, 3, 3, 3, 3, 3]), 3, 50, 5),
            (np.array([9, 9, 0, 0, 0]), 9, 2, 5),
        ],
    )
    def test_harmonic_mean_equals_closed_form_capped(
        self, labels, query, rq, k
    ) -> None:
        """Capped F1 == 2*r_K / (K + min(K, R_q))."""
        labels = np.asarray(labels)
        r_k = int(np.count_nonzero(labels[:k] == query))
        denom = k + min(k, rq)
        closed = 2.0 * r_k / denom if rq > 0 else 0.0
        via_hm = f1_at_k(labels, query, rq, k=k, mode="capped")
        assert via_hm == pytest.approx(closed)

    def test_no_relevant_retrieved_is_zero(self) -> None:
        labels = np.array([0, 0, 0, 0, 0], dtype=np.int64)
        assert f1_at_k(labels, 1, total_relevant=8, k=5) == 0.0

    def test_zero_total_relevant_is_zero(self) -> None:
        assert f1_at_k(WORKED_LABELS, 1, total_relevant=0, k=5) == 0.0


# ---------------------------------------------------------------------------
# Average Precision
# ---------------------------------------------------------------------------
class TestAveragePrecision:
    def test_perfect_ranking_ap_is_one(self) -> None:
        # All R_q relevants ranked first -> AP = 1.0.
        labels = np.array([4, 4, 4, 0, 0], dtype=np.int64)
        assert average_precision(labels, 4, total_relevant=3) == pytest.approx(1.0)

    def test_hand_computed_ap(self) -> None:
        # ranked relevance [1,0,1,0,1], R_q = 3.
        # precisions at relevant ranks: rank1 -> 1/1, rank3 -> 2/3, rank5 -> 3/5
        # AP = (1 + 2/3 + 3/5) / 3
        labels = np.array([1, 0, 1, 0, 1], dtype=np.int64)
        expected = (1.0 + 2.0 / 3.0 + 3.0 / 5.0) / 3.0
        assert average_precision(labels, 1, total_relevant=3) == pytest.approx(
            expected
        )

    def test_ap_penalises_unretrieved_relevants(self) -> None:
        # Only 1 of 4 relevants retrieved (at rank 1): AP = (1/1)/4 = 0.25.
        labels = np.array([1, 0, 0, 0, 0], dtype=np.int64)
        assert average_precision(labels, 1, total_relevant=4) == pytest.approx(
            0.25
        )

    def test_ap_zero_when_no_relevant(self) -> None:
        labels = np.array([0, 0, 0], dtype=np.int64)
        assert average_precision(labels, 1, total_relevant=5) == 0.0

    def test_ap_zero_total_relevant(self) -> None:
        assert average_precision(WORKED_LABELS, 1, total_relevant=0) == 0.0

    def test_ap_truncation_k(self) -> None:
        # Truncating to k=1 with [1,0,1,...] -> only first relevant counts.
        labels = np.array([1, 0, 1, 0, 1], dtype=np.int64)
        # AP@1 = (1/1)/3 = 1/3
        assert average_precision(labels, 1, total_relevant=3, k=1) == pytest.approx(
            1.0 / 3.0
        )


# ---------------------------------------------------------------------------
# nDCG@K
# ---------------------------------------------------------------------------
class TestNDCGAtK:
    def test_perfect_ranking_ndcg_is_one(self) -> None:
        labels = np.array([2, 2, 2, 0, 0], dtype=np.int64)
        assert ndcg_at_k(labels, 2, total_relevant=3, k=5) == pytest.approx(1.0)

    def test_hand_computed_ndcg(self) -> None:
        # ranked relevance [0,1,1], R_q = 2, k = 3.
        # DCG = 0/log2(2) + 1/log2(3) + 1/log2(4)
        #     = 0 + 1/1.5849625 + 1/2 = 0.6309298 + 0.5 = 1.1309298
        # IDCG (2 relevants ideal) = 1/log2(2) + 1/log2(3) = 1 + 0.6309298
        #     = 1.6309298
        labels = np.array([0, 1, 1], dtype=np.int64)
        dcg = 1.0 / math.log2(3) + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        assert ndcg_at_k(labels, 1, total_relevant=2, k=3) == pytest.approx(
            dcg / idcg
        )

    def test_ndcg_uses_rq_for_ideal(self) -> None:
        # Only 1 relevant retrieved but R_q = 3: ideal DCG uses 3 ones.
        labels = np.array([1, 0, 0], dtype=np.int64)
        dcg = 1.0 / math.log2(2)  # single hit at rank 1
        idcg = (
            1.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4)
        )
        assert ndcg_at_k(labels, 1, total_relevant=3, k=3) == pytest.approx(
            dcg / idcg
        )

    def test_ndcg_zero_when_no_relevant(self) -> None:
        labels = np.array([0, 0, 0], dtype=np.int64)
        assert ndcg_at_k(labels, 1, total_relevant=0, k=3) == 0.0

    def test_ndcg_graded_gains(self) -> None:
        # Graded relevance via explicit gains.
        gains = [3.0, 2.0, 0.0, 1.0]
        dcg = (
            3.0 / math.log2(2)
            + 2.0 / math.log2(3)
            + 0.0 / math.log2(4)
            + 1.0 / math.log2(5)
        )
        ideal = sorted(gains, reverse=True)
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        assert ndcg_at_k(k=4, gains=gains) == pytest.approx(dcg / idcg)


# ---------------------------------------------------------------------------
# Reciprocal Rank
# ---------------------------------------------------------------------------
class TestReciprocalRank:
    def test_first_relevant_at_rank_1(self) -> None:
        labels = np.array([5, 0, 0], dtype=np.int64)
        assert reciprocal_rank(labels, 5) == 1.0

    def test_first_relevant_at_rank_3(self) -> None:
        labels = np.array([0, 0, 5, 0], dtype=np.int64)
        assert reciprocal_rank(labels, 5) == pytest.approx(1.0 / 3.0)

    def test_no_relevant_is_zero(self) -> None:
        labels = np.array([0, 0, 0], dtype=np.int64)
        assert reciprocal_rank(labels, 5) == 0.0

    def test_truncation_excludes_late_hit(self) -> None:
        labels = np.array([0, 0, 0, 5], dtype=np.int64)
        # First (only) hit at rank 4; truncate to k=3 -> no hit -> 0.
        assert reciprocal_rank(labels, 5, k=3) == 0.0


# ---------------------------------------------------------------------------
# Relevance helper
# ---------------------------------------------------------------------------
class TestRelevanceFromLabels:
    def test_mask_matches_equality(self) -> None:
        labels = np.array([1, 2, 1, 3, 1], dtype=np.int64)
        mask = relevance_from_labels(labels, 1)
        assert mask.dtype == bool
        np.testing.assert_array_equal(mask, [True, False, True, False, True])


# ---------------------------------------------------------------------------
# Vectorised batch helpers
# ---------------------------------------------------------------------------
class TestBatchF1AtK:
    def test_matches_scalar_loop(self) -> None:
        rng = np.random.default_rng(0)
        q, r = 25, 10
        labels = rng.integers(0, 4, size=(q, r))
        q_labels = rng.integers(0, 4, size=q)
        r_q = rng.integers(1, 12, size=q)
        batch = batch_f1_at_k(labels, q_labels, r_q, k=5, mode="raw")
        for i in range(q):
            scalar = f1_at_k(labels[i], int(q_labels[i]), int(r_q[i]), k=5)
            assert batch[i] == pytest.approx(scalar)

    def test_capped_mode_matches_scalar(self) -> None:
        rng = np.random.default_rng(1)
        q, r = 15, 10
        labels = rng.integers(0, 3, size=(q, r))
        q_labels = rng.integers(0, 3, size=q)
        r_q = rng.integers(1, 8, size=q)
        batch = batch_f1_at_k(labels, q_labels, r_q, k=10, mode="capped")
        for i in range(q):
            scalar = f1_at_k(
                labels[i], int(q_labels[i]), int(r_q[i]), k=10, mode="capped"
            )
            assert batch[i] == pytest.approx(scalar)

    def test_zero_rq_rows_are_zero(self) -> None:
        labels = np.array([[1, 1, 1, 1, 1]], dtype=np.int64)
        out = batch_f1_at_k(labels, np.array([1]), np.array([0]), k=5)
        assert out[0] == 0.0

    def test_worked_example_row(self) -> None:
        out = batch_f1_at_k(
            WORKED_LABELS[None, :],
            np.array([WORKED_QUERY_LABEL]),
            np.array([WORKED_RQ]),
            k=5,
            mode="raw",
        )
        assert out[0] == pytest.approx(8.0 / 13.0)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            batch_f1_at_k(
                np.zeros((3, 5)), np.array([1, 2]), np.array([1, 2, 3]), k=5
            )


class TestMeanMetrics:
    def test_keys_and_substitution(self) -> None:
        labels = WORKED_LABELS[None, :]
        res = mean_metrics(
            labels, np.array([1]), np.array([WORKED_RQ]), k=5, mode="raw"
        )
        assert set(res) == {"P@5", "R@5", "F1@5", "mAP", "nDCG@5", "MRR", "n_queries"}
        assert res["P@5"] == pytest.approx(0.8)
        assert res["R@5"] == pytest.approx(0.5)
        assert res["F1@5"] == pytest.approx(8.0 / 13.0)
        assert res["MRR"] == pytest.approx(1.0)  # first item relevant
        assert res["n_queries"] == 1.0

    def test_macro_average_two_queries(self) -> None:
        # Query A: [1,1,1,0,1] q=1 R_q=4 -> F1@5 = 2*4/(5+4)=8/9
        # Query B: [0,0,2,0,0] q=2 R_q=1 -> F1@5 = 2*1/(5+1)=1/3
        labels = np.array([[1, 1, 1, 0, 1], [0, 0, 2, 0, 0]], dtype=np.int64)
        q_labels = np.array([1, 2])
        r_q = np.array([4, 1])
        res = mean_metrics(labels, q_labels, r_q, k=5, mode="raw")
        expected_f1 = ((8.0 / 9.0) + (1.0 / 3.0)) / 2.0
        assert res["F1@5"] == pytest.approx(expected_f1)

    def test_empty_batch(self) -> None:
        res = mean_metrics(
            np.zeros((0, 5), dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            k=5,
        )
        assert res["n_queries"] == 0.0
        assert res["F1@5"] == 0.0


# ---------------------------------------------------------------------------
# Property: F1 is the exact harmonic mean of P and R for a random battery.
# ---------------------------------------------------------------------------
def test_f1_is_harmonic_mean_property() -> None:
    rng = np.random.default_rng(42)
    for _ in range(200):
        k = int(rng.integers(1, 12))
        n = int(rng.integers(1, 15))
        labels = rng.integers(0, 3, size=n)
        query = int(rng.integers(0, 3))
        r_q = int(rng.integers(0, 20))
        p = precision_at_k(labels, query, k=k)
        r = recall_at_k(labels, query, r_q, k=k, mode="raw")
        f1 = f1_at_k(labels, query, r_q, k=k, mode="raw")
        if p + r == 0:
            assert f1 == 0.0
        else:
            assert f1 == pytest.approx(2 * p * r / (p + r))
