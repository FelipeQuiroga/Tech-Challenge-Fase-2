"""Carregador concreto para o dataset RetailRocket."""

from __future__ import annotations

import pandas as pd

from ecommerce_recommender.data_loaders.base import DataLoaderStrategy
from ecommerce_recommender.exceptions import DatasetPathNotFoundError

_EVENTS_FILENAME = "events.csv"
_EVENTS_DTYPES: dict[str, str] = {
    "timestamp": "int64",
    "visitorid": "int32",
    "event": "category",
    "itemid": "int32",
    "transactionid": "Int64",
}


class RetailRocketDataLoader(DataLoaderStrategy):
    """Carregador para o dataset RetailRocket de e-commerce.

    Lê o arquivo de eventos de navegação (``events.csv``), que contém as
    interações de usuários com produtos: visualizações, adições ao carrinho
    e transações.
    """

    def load(self) -> pd.DataFrame:
        """Carrega os eventos de navegação do dataset RetailRocket.

        Returns:
            DataFrame com as colunas ``timestamp``, ``visitorid``, ``event``,
            ``itemid`` e ``transactionid``, uma linha por evento.

        Raises:
            DatasetPathNotFoundError: Se ``events.csv`` não existir dentro de
                ``source_path``.
        """
        events_path = self.source_path / _EVENTS_FILENAME
        if not events_path.exists():
            raise DatasetPathNotFoundError(
                f"Arquivo de eventos não encontrado: {events_path}"
            )
        return pd.read_csv(events_path, dtype=_EVENTS_DTYPES)
