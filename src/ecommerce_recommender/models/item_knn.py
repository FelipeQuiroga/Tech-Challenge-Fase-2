"""Baseline de recomendação por similaridade item-item (Scikit-Learn)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from ecommerce_recommender.models.base import RecommenderStrategy


class ItemKnnRecommender(RecommenderStrategy):
    """Baseline de filtragem colaborativa item-item.

    Pontua um item candidato pela maior similaridade de cosseno
    (``sklearn.metrics.pairwise.cosine_similarity``) entre ele e os itens
    com os quais o usuário já interagiu no treino.
    """

    def __init__(self, n_users: int, n_items: int) -> None:
        """Inicializa o baseline com o tamanho do vocabulário global.

        Args:
            n_users: Quantidade total de usuários no vocabulário global.
            n_items: Quantidade total de itens no vocabulário global.
        """
        self._n_users = n_users
        self._n_items = n_items
        self._item_user_matrix: sparse.csr_matrix | None = None
        self._user_items: dict[int, list[int]] = {}

    def fit(
        self, train_events: pd.DataFrame, val_events: pd.DataFrame | None = None
    ) -> None:
        """Constrói a matriz esparsa item-usuário a partir do treino.

        Args:
            train_events: Eventos de treino, com colunas ``user_index`` e
                ``item_index``.
            val_events: Ignorado — este baseline não usa early stopping.
        """
        interactions = train_events[["user_index", "item_index"]].drop_duplicates()
        self._item_user_matrix = sparse.coo_matrix(
            (
                np.ones(len(interactions), dtype=np.float32),
                (interactions["item_index"], interactions["user_index"]),
            ),
            shape=(self._n_items, self._n_users),
        ).tocsr()
        self._user_items = (
            interactions.groupby("user_index")["item_index"].apply(list).to_dict()
        )

    def score_items(self, user_index: int, item_indices: np.ndarray) -> np.ndarray:
        """Pontua candidatos pela similaridade máxima a um item conhecido.

        Args:
            user_index: Índice do usuário.
            item_indices: Itens candidatos a pontuar.

        Returns:
            Similaridade de cosseno máxima entre cada candidato e os itens
            com os quais o usuário interagiu no treino (0 se desconhecido).
        """
        known_items = self._user_items.get(user_index, [])
        if not known_items or self._item_user_matrix is None:
            return np.zeros(len(item_indices))
        candidate_rows = self._item_user_matrix[item_indices]
        known_rows = self._item_user_matrix[known_items]
        similarity = cosine_similarity(candidate_rows, known_rows)
        return similarity.max(axis=1)
