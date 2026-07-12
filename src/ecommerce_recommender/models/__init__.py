"""Estratégias de recomendação (Strategy) e fábrica de seleção (Factory)."""

from ecommerce_recommender.models.base import RecommenderStrategy
from ecommerce_recommender.models.factory import available_models, create_recommender
from ecommerce_recommender.models.item_knn import ItemKnnRecommender
from ecommerce_recommender.models.ncf import NCFRecommender

__all__ = [
    "ItemKnnRecommender",
    "NCFRecommender",
    "RecommenderStrategy",
    "available_models",
    "create_recommender",
]
