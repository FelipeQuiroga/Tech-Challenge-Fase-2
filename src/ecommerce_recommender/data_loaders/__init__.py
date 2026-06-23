"""Carregadores de dataset e fábrica de seleção por nome."""

from ecommerce_recommender.data_loaders.base import DataLoaderStrategy
from ecommerce_recommender.data_loaders.factory import (
    available_loaders,
    create_data_loader,
)
from ecommerce_recommender.data_loaders.retailrocket import RetailRocketDataLoader

__all__ = [
    "DataLoaderStrategy",
    "RetailRocketDataLoader",
    "available_loaders",
    "create_data_loader",
]
