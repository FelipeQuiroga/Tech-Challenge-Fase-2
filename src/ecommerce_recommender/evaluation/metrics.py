"""Métricas de ranking para avaliação de recomendação top-K."""

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Fração dos top-k recomendados que são relevantes.

    Args:
        recommended: Itens recomendados, em ordem de relevância decrescente.
        relevant: Conjunto de itens relevantes (verdade fundamental).
        k: Tamanho do corte do ranking.

    Returns:
        Precisão no corte ``k``, entre 0 e 1.
    """
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Fração dos itens relevantes capturados nos top-k recomendados.

    Args:
        recommended: Itens recomendados, em ordem de relevância decrescente.
        relevant: Conjunto de itens relevantes (verdade fundamental).
        k: Tamanho do corte do ranking.

    Returns:
        Recall no corte ``k``, entre 0 e 1.
    """
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Indica se ao menos um item relevante aparece nos top-k recomendados.

    Args:
        recommended: Itens recomendados, em ordem de relevância decrescente.
        relevant: Conjunto de itens relevantes (verdade fundamental).
        k: Tamanho do corte do ranking.

    Returns:
        ``1.0`` se houve ao menos um acerto, ``0.0`` caso contrário.
    """
    top_k = recommended[:k]
    return 1.0 if any(item in relevant for item in top_k) else 0.0


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Ganho cumulativo descontado normalizado (NDCG) no corte ``k``.

    Penaliza itens relevantes posicionados mais abaixo no ranking.

    Args:
        recommended: Itens recomendados, em ordem de relevância decrescente.
        relevant: Conjunto de itens relevantes (verdade fundamental).
        k: Tamanho do corte do ranking.

    Returns:
        NDCG no corte ``k``, entre 0 e 1.
    """
    top_k = recommended[:k]
    dcg = sum(
        1.0 / math.log2(position + 2)
        for position, item in enumerate(top_k)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(position + 2) for position in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
