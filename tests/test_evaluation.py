"""Testes das métricas de ranking e da agregação por usuário."""

import math

from ecommerce_recommender.evaluation import (
    evaluate_rankings,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k_counts_hits_in_top_k() -> None:
    assert precision_at_k([1, 2, 3, 4], {2, 4, 99}, k=2) == 0.5


def test_precision_at_k_empty_recommendations_is_zero() -> None:
    assert precision_at_k([], {1}, k=5) == 0.0


def test_recall_at_k_divides_by_relevant_count() -> None:
    assert recall_at_k([1, 2, 3], {2, 3, 4}, k=2) == 1 / 3


def test_recall_at_k_no_relevant_items_is_zero() -> None:
    assert recall_at_k([1, 2], set(), k=2) == 0.0


def test_hit_rate_at_k_true_when_any_hit() -> None:
    assert hit_rate_at_k([1, 2, 3], {3}, k=3) == 1.0
    assert hit_rate_at_k([1, 2, 3], {99}, k=3) == 0.0


def test_ndcg_at_k_rewards_earlier_hits() -> None:
    early_hit = ndcg_at_k([1, 2, 3], {1}, k=3)
    late_hit = ndcg_at_k([2, 3, 1], {1}, k=3)
    assert early_hit == 1.0
    assert 0 < late_hit < early_hit


def test_ndcg_at_k_matches_manual_computation() -> None:
    result = ndcg_at_k([9, 1, 9], {1}, k=3)
    expected_dcg = 1.0 / math.log2(3)
    assert math.isclose(result, expected_dcg)


def test_evaluate_rankings_averages_across_users() -> None:
    recommendations = {1: [10, 20], 2: [30, 40]}
    relevant_items = {1: {10}, 2: {99}}

    scores = evaluate_rankings(recommendations, relevant_items, k=2)

    assert scores["hit_rate"] == 0.5
    assert scores["precision"] == 0.25
    assert set(scores) == {"precision", "recall", "ndcg", "hit_rate"}


def test_evaluate_rankings_empty_input_returns_zeros() -> None:
    scores = evaluate_rankings({}, {}, k=5)
    assert all(value == 0.0 for value in scores.values())
