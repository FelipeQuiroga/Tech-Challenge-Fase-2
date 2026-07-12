"""Fábrica de recomendadores por nome (Factory Pattern)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ecommerce_recommender.exceptions import UnsupportedModelError
from ecommerce_recommender.models.base import RecommenderStrategy
from ecommerce_recommender.models.item_knn import ItemKnnRecommender
from ecommerce_recommender.models.ncf import NCFRecommender

_MODEL_REGISTRY: dict[str, Callable[..., RecommenderStrategy]] = {
    "ncf": NCFRecommender,
    "item_knn": ItemKnnRecommender,
}


def available_models() -> tuple[str, ...]:
    """Lista os nomes de modelos suportados.

    Returns:
        Tupla com os nomes registrados na fábrica.
    """
    return tuple(_MODEL_REGISTRY)


def create_recommender(
    name: str, n_users: int, n_items: int, **kwargs: Any
) -> RecommenderStrategy:
    """Cria uma estratégia de recomendação a partir do nome.

    Args:
        name: Nome do modelo (ver :func:`available_models`).
        n_users: Quantidade total de usuários no vocabulário global.
        n_items: Quantidade total de itens no vocabulário global.
        **kwargs: Hiperparâmetros adicionais específicos do modelo.

    Returns:
        Instância concreta de :class:`RecommenderStrategy`.

    Raises:
        UnsupportedModelError: Se ``name`` não for reconhecido.
    """
    model_cls = _MODEL_REGISTRY.get(name)
    if model_cls is None:
        supported = ", ".join(available_models())
        raise UnsupportedModelError(
            f"Modelo não suportado: {name!r}. Suportados: {supported}."
        )
    return model_cls(n_users=n_users, n_items=n_items, **kwargs)
