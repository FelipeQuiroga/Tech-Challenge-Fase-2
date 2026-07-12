"""Interface Strategy comum aos modelos de recomendação."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class RecommenderStrategy(ABC):
    """Interface comum para estratégias de recomendação treináveis.

    Cada implementação concreta encapsula um algoritmo de recomendação
    diferente (rede neural, similaridade item-item, etc.), permitindo
    treiná-las e avaliá-las de forma intercambiável. ``recommend`` é um
    método-template construído sobre ``score_items`` (abstrato).
    """

    @abstractmethod
    def fit(
        self, train_events: pd.DataFrame, val_events: pd.DataFrame | None = None
    ) -> None:
        """Treina o modelo a partir dos eventos de treino.

        Args:
            train_events: DataFrame com, no mínimo, as colunas
                ``user_index`` e ``item_index``.
            val_events: Eventos de validação, para modelos que suportam
                early stopping. Ignorado pelos modelos que não usam.
        """
        raise NotImplementedError

    @abstractmethod
    def score_items(self, user_index: int, item_indices: np.ndarray) -> np.ndarray:
        """Pontua um conjunto de itens candidatos para um usuário.

        Args:
            user_index: Índice contíguo do usuário.
            item_indices: Itens candidatos a pontuar.

        Returns:
            Array de pontuações (maior = mais relevante), na mesma ordem de
            ``item_indices``.
        """
        raise NotImplementedError

    def recommend(self, user_index: int, item_indices: np.ndarray, k: int) -> list[int]:
        """Recomenda os ``k`` itens mais relevantes dentre os candidatos.

        Args:
            user_index: Índice contíguo do usuário.
            item_indices: Itens candidatos a ranquear.
            k: Quantidade de itens a recomendar.

        Returns:
            Os ``k`` itens candidatos mais bem pontuados, em ordem
            decrescente de relevância.
        """
        scores = self.score_items(user_index, item_indices)
        top_k_positions = np.argsort(-scores)[:k]
        return [int(item_indices[position]) for position in top_k_positions]
