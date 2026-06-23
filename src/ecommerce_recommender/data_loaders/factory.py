"""Fábrica para seleção de carregadores de dataset pelo nome (Factory Pattern)."""

from __future__ import annotations

from pathlib import Path

from ecommerce_recommender.data_loaders.base import DataLoaderStrategy
from ecommerce_recommender.data_loaders.retailrocket import RetailRocketDataLoader
from ecommerce_recommender.exceptions import UnsupportedDatasetError

_LOADER_REGISTRY: dict[str, type[DataLoaderStrategy]] = {
    "retailrocket": RetailRocketDataLoader,
}


def available_loaders() -> tuple[str, ...]:
    """Lista os nomes de datasets suportados.

    Returns:
        Tupla com os nomes registrados na fábrica.
    """
    return tuple(_LOADER_REGISTRY)


def create_data_loader(name: str, source_path: Path) -> DataLoaderStrategy:
    """Cria um carregador de dataset a partir do nome.

    Args:
        name: Nome do dataset (insensível a maiúsculas/minúsculas).
        source_path: Diretório com os arquivos brutos do dataset.

    Returns:
        Instância concreta de :class:`DataLoaderStrategy`.

    Raises:
        UnsupportedDatasetError: Se o nome do dataset não for suportado.
    """
    loader_cls = _LOADER_REGISTRY.get(name.lower())
    if loader_cls is None:
        supported = ", ".join(available_loaders())
        raise UnsupportedDatasetError(
            f"Dataset não suportado: {name!r}. Suportados: {supported}."
        )
    return loader_cls(source_path)
