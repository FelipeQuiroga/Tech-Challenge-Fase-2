"""Agregação das métricas de ranking sobre um conjunto de usuários."""

from __future__ import annotations

from ecommerce_recommender.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

_METRIC_FUNCTIONS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "hit_rate": hit_rate_at_k,
}


def evaluate_rankings(
    recommendations: dict[int, list[int]],
    relevant_items: dict[int, set[int]],
    k: int,
) -> dict[str, float]:
    """Calcula a média de cada métrica de ranking sobre todos os usuários.

    Args:
        recommendations: Itens recomendados por usuário, em ordem de
            relevância decrescente.
        relevant_items: Itens relevantes (verdade fundamental) por usuário.
        k: Tamanho do corte do ranking.

    Returns:
        Dicionário com a média de cada métrica (``precision``, ``recall``,
        ``ndcg``, ``hit_rate``) sobre os usuários avaliados.
    """
    totals = dict.fromkeys(_METRIC_FUNCTIONS, 0.0)
    n_users = len(recommendations)
    for user, recommended in recommendations.items():
        relevant = relevant_items.get(user, set())
        for name, metric_fn in _METRIC_FUNCTIONS.items():
            totals[name] += metric_fn(recommended, relevant, k)
    return {name: total / n_users if n_users else 0.0 for name, total in totals.items()}
