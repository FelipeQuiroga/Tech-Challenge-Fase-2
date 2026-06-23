"""Carregador concreto para o dataset RetailRocket."""

from __future__ import annotations

from ecommerce_recommender.data_loaders.base import DataLoaderStrategy


class RetailRocketDataLoader(DataLoaderStrategy):
    """Carregador para o dataset RetailRocket de e-commerce.

    Esta é uma implementação inicial: o processamento dos arquivos ainda não
    foi implementado e será adicionado em etapas futuras.
    """

    def load(self) -> object:
        """Carrega o dataset RetailRocket.

        Returns:
            Os dados carregados (a ser implementado).

        Raises:
            NotImplementedError: O processamento ainda não foi implementado.
        """
        raise NotImplementedError(
            "O processamento do dataset RetailRocket será implementado "
            "em uma etapa futura."
        )
