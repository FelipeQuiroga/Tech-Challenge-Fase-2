"""Modelo de recomendação neural: embeddings de usuário/item + MLP (NCF)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from ecommerce_recommender.models.base import RecommenderStrategy

_POSITIVE_LABEL = 1.0
_NEGATIVE_LABEL = 0.0


class _NCFModule(nn.Module):
    """Rede neural: embeddings de usuário/item concatenados, seguidos de MLP."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        hidden_dims: tuple[int, ...],
    ) -> None:
        """Constrói as camadas de embedding e a MLP.

        Args:
            n_users: Quantidade total de usuários no vocabulário global.
            n_items: Quantidade total de itens no vocabulário global.
            embedding_dim: Dimensão dos embeddings de usuário e item.
            hidden_dims: Tamanhos das camadas ocultas da MLP.
        """
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.mlp = self._build_mlp(embedding_dim * 2, hidden_dims)

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(current_dim, hidden_dim), nn.ReLU()]
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """Calcula o logit de interação para pares (usuário, item).

        Args:
            user_idx: Índices de usuário, shape ``(batch,)``.
            item_idx: Índices de item, shape ``(batch,)``.

        Returns:
            Logits (pré-sigmoid) de interação, shape ``(batch,)``.
        """
        features = torch.cat(
            [self.user_embedding(user_idx), self.item_embedding(item_idx)], dim=1
        )
        return self.mlp(features).squeeze(-1)


class NCFRecommender(RecommenderStrategy):
    """Recomendador neural (Neural Collaborative Filtering) treinado com PyTorch.

    Combina embeddings de usuário e item com uma MLP, treinada como
    classificação binária (interagiu vs. amostra negativa) com amostragem
    negativa por época e early stopping na perda de validação.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        hidden_dims: tuple[int, ...] = (64, 32),
        negatives_per_positive: int = 4,
        batch_size: int = 4096,
        max_epochs: int = 10,
        patience: int = 2,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        """Cria o modelo e fixa a semente de reprodutibilidade.

        Args:
            n_users: Quantidade total de usuários no vocabulário global.
            n_items: Quantidade total de itens no vocabulário global.
            embedding_dim: Dimensão dos embeddings de usuário e item.
            hidden_dims: Tamanhos das camadas ocultas da MLP.
            negatives_per_positive: Negativos amostrados por interação
                positiva, a cada época de treino.
            batch_size: Tamanho do lote de treino/avaliação.
            max_epochs: Número máximo de épocas.
            patience: Épocas sem melhora na perda de validação antes de
                parar (early stopping).
            learning_rate: Taxa de aprendizado do otimizador Adam.
            seed: Semente para reprodutibilidade.
        """
        self._n_items = n_items
        self._hidden_dims = hidden_dims
        self._negatives_per_positive = negatives_per_positive
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._patience = patience
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self._model = _NCFModule(n_users, n_items, embedding_dim, hidden_dims)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        self._criterion = nn.BCEWithLogitsLoss()
        self.history: list[dict[str, float | None]] = []

    def fit(
        self, train_events: pd.DataFrame, val_events: pd.DataFrame | None = None
    ) -> None:
        """Treina o modelo, com early stopping na perda de validação.

        Args:
            train_events: Eventos de treino, com colunas ``user_index`` e
                ``item_index``.
            val_events: Eventos de validação (opcional). Se informado,
                habilita early stopping.
        """
        train_users = train_events["user_index"].to_numpy()
        train_items = train_events["item_index"].to_numpy()
        val_tensors = (
            self._build_eval_tensors(val_events) if val_events is not None else None
        )

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        for epoch in range(self._max_epochs):
            train_loss = self._train_one_epoch(train_users, train_items)
            val_loss = (
                self._evaluate_loss(val_tensors) if val_tensors is not None else None
            )
            self.history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )
            if val_loss is None:
                continue
            if val_loss < best_val_loss:
                best_val_loss, epochs_without_improvement = val_loss, 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self._patience:
                    break

    def _train_one_epoch(self, users: np.ndarray, items: np.ndarray) -> float:
        """Executa uma época de treino com amostragem negativa fresca.

        Args:
            users: Índices de usuário das interações positivas de treino.
            items: Índices de item das interações positivas de treino.

        Returns:
            Perda média (BCE) da época.
        """
        all_users, all_items, all_labels = self._build_labeled_tensors(users, items)
        self._model.train()
        total_loss = 0.0
        for batch_users, batch_items, batch_labels in self._iterate_batches(
            all_users, all_items, all_labels, shuffle=True
        ):
            self._optimizer.zero_grad()
            logits = self._model(batch_users, batch_items)
            loss = self._criterion(logits, batch_labels)
            loss.backward()
            self._optimizer.step()
            total_loss += loss.item() * len(batch_labels)
        return total_loss / len(all_labels)

    def _iterate_batches(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        labels: torch.Tensor,
        shuffle: bool,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Itera em lotes sobre tensores pré-carregados, via fatiamento vetorizado.

        Evita o overhead por amostra do ``DataLoader``/``TensorDataset``
        padrão, relevante em datasets de milhões de linhas.

        Args:
            users: Índices de usuário de todas as amostras da época.
            items: Índices de item de todas as amostras da época.
            labels: Rótulos (0/1) de todas as amostras da época.
            shuffle: Se ``True``, embaralha a ordem antes de fatiar em lotes.

        Yields:
            Lotes de ``(usuários, itens, rótulos)`` de até ``batch_size``.
        """
        n = len(labels)
        if shuffle:
            order = torch.from_numpy(self._rng.permutation(n))
        else:
            order = torch.arange(n)
        for start in range(0, n, self._batch_size):
            batch_idx = order[start : start + self._batch_size]
            yield users[batch_idx], items[batch_idx], labels[batch_idx]

    def _build_eval_tensors(
        self, events: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Monta os tensores de avaliação (positivos + negativos rotulados).

        Args:
            events: Eventos com colunas ``user_index`` e ``item_index``.

        Returns:
            Tupla ``(usuários, itens, rótulos)``.
        """
        return self._build_labeled_tensors(
            events["user_index"].to_numpy(), events["item_index"].to_numpy()
        )

    def _build_labeled_tensors(
        self, users: np.ndarray, items: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combina interações positivas com negativos amostrados uniformemente.

        Args:
            users: Índices de usuário das interações positivas.
            items: Índices de item das interações positivas.

        Returns:
            Tensores ``(usuários, itens, rótulos)`` prontos para batching.
        """
        n_neg = self._negatives_per_positive
        neg_users = np.repeat(users, n_neg)
        neg_items = self._rng.integers(0, self._n_items, size=neg_users.shape[0])
        all_users = np.concatenate([users, neg_users])
        all_items = np.concatenate([items, neg_items])
        labels = np.concatenate(
            [
                np.full(len(users), _POSITIVE_LABEL),
                np.full(len(neg_users), _NEGATIVE_LABEL),
            ]
        ).astype(np.float32)
        return (
            torch.from_numpy(all_users.astype(np.int64)),
            torch.from_numpy(all_items.astype(np.int64)),
            torch.from_numpy(labels),
        )

    def _evaluate_loss(
        self, tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> float:
        """Calcula a perda média (BCE) sobre tensores pré-carregados.

        Args:
            tensors: Tupla ``(usuários, itens, rótulos)`` de avaliação.

        Returns:
            Perda média sobre todos os exemplos.
        """
        users, items, labels = tensors
        self._model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_users, batch_items, batch_labels in self._iterate_batches(
                users, items, labels, shuffle=False
            ):
                logits = self._model(batch_users, batch_items)
                loss = self._criterion(logits, batch_labels)
                total_loss += loss.item() * len(batch_labels)
        return total_loss / len(labels)

    def score_items(self, user_index: int, item_indices: np.ndarray) -> np.ndarray:
        """Pontua itens candidatos para um usuário via o modelo treinado.

        Args:
            user_index: Índice do usuário.
            item_indices: Itens candidatos a pontuar.

        Returns:
            Probabilidades de interação (sigmoid), na ordem de
            ``item_indices``.
        """
        self._model.eval()
        with torch.no_grad():
            users = torch.full((len(item_indices),), user_index, dtype=torch.long)
            items = torch.from_numpy(item_indices.astype(np.int64))
            logits = self._model(users, items)
        return torch.sigmoid(logits).numpy()

    def save(self, path: Path) -> None:
        """Salva os pesos e hiperparâmetros do modelo em disco.

        Args:
            path: Caminho do arquivo de checkpoint (``.pt``).
        """
        checkpoint: dict[str, Any] = {
            "state_dict": self._model.state_dict(),
            "n_users": self._model.user_embedding.num_embeddings,
            "n_items": self._model.item_embedding.num_embeddings,
            "embedding_dim": self._model.user_embedding.embedding_dim,
            "hidden_dims": self._hidden_dims,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: Path) -> NCFRecommender:
        """Carrega um modelo previamente salvo com :meth:`save`.

        Args:
            path: Caminho do arquivo de checkpoint (``.pt``).

        Returns:
            Instância de :class:`NCFRecommender` com os pesos carregados.
        """
        checkpoint = torch.load(path, weights_only=True)
        instance = cls(
            n_users=checkpoint["n_users"],
            n_items=checkpoint["n_items"],
            embedding_dim=checkpoint["embedding_dim"],
            hidden_dims=tuple(checkpoint["hidden_dims"]),
        )
        instance._model.load_state_dict(checkpoint["state_dict"])
        return instance
