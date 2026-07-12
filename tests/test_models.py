"""Testes das estratégias de recomendação e da fábrica de seleção."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecommerce_recommender.exceptions import UnsupportedModelError
from ecommerce_recommender.models import (
    ItemKnnRecommender,
    NCFRecommender,
    available_models,
    create_recommender,
)
from ecommerce_recommender.models.base import RecommenderStrategy


class _ConstantRecommender(RecommenderStrategy):
    """Estratégia fake que pontua cada item pelo próprio índice.

    Usada para testar o método-template ``recommend`` isoladamente.
    """

    def fit(
        self, train_events: pd.DataFrame, val_events: pd.DataFrame | None = None
    ) -> None:
        pass

    def score_items(self, user_index: int, item_indices: np.ndarray) -> np.ndarray:
        return item_indices.astype(float)


def _toy_events(n_users: int = 6, n_items: int = 6, n_rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "user_index": rng.integers(0, n_users, size=n_rows),
            "item_index": rng.integers(0, n_items, size=n_rows),
        }
    )


def test_recommend_template_method_orders_by_score_descending() -> None:
    recommender = _ConstantRecommender()
    result = recommender.recommend(
        user_index=0, item_indices=np.array([3, 1, 5, 2]), k=2
    )
    assert result == [5, 3]


def test_available_models_lists_registered_names() -> None:
    assert set(available_models()) == {"ncf", "item_knn"}


def test_create_recommender_unsupported_name_raises() -> None:
    with pytest.raises(UnsupportedModelError):
        create_recommender("unknown", n_users=1, n_items=1)


def test_create_recommender_builds_requested_model() -> None:
    model = create_recommender("item_knn", n_users=6, n_items=6)
    assert isinstance(model, ItemKnnRecommender)


def test_item_knn_recommends_items_similar_to_known_ones() -> None:
    events = pd.DataFrame(
        {
            "user_index": [0, 0, 1, 1, 2],
            "item_index": [0, 1, 0, 1, 2],
        }
    )
    model = ItemKnnRecommender(n_users=3, n_items=3)
    model.fit(events)

    scores = model.score_items(user_index=0, item_indices=np.array([0, 1, 2]))

    assert scores[2] == 0.0
    assert scores[0] > 0.0 and scores[1] > 0.0


def test_item_knn_unknown_user_returns_zero_scores() -> None:
    model = ItemKnnRecommender(n_users=3, n_items=3)
    model.fit(pd.DataFrame({"user_index": [0], "item_index": [0]}))

    scores = model.score_items(user_index=99, item_indices=np.array([0, 1]))

    assert list(scores) == [0.0, 0.0]


def test_ncf_trains_and_scores_all_candidates(tmp_path: Path) -> None:
    events = _toy_events()
    train, val = events.iloc[:40], events.iloc[40:]

    model = NCFRecommender(
        n_users=6,
        n_items=6,
        embedding_dim=4,
        hidden_dims=(8,),
        max_epochs=2,
        batch_size=16,
    )
    model.fit(train, val_events=val)

    scores = model.score_items(user_index=0, item_indices=np.array([0, 1, 2, 3]))
    assert scores.shape == (4,)
    assert len(model.history) == 2
    assert model.history[0]["val_loss"] is not None


def test_ncf_save_and_load_roundtrip_preserves_scores(tmp_path: Path) -> None:
    events = _toy_events()
    model = NCFRecommender(
        n_users=6, n_items=6, embedding_dim=4, hidden_dims=(8,), max_epochs=1
    )
    model.fit(events)
    candidates = np.array([0, 1, 2, 3, 4])
    original_scores = model.score_items(user_index=1, item_indices=candidates)

    checkpoint_path = tmp_path / "ncf.pt"
    model.save(checkpoint_path)
    loaded = NCFRecommender.load(checkpoint_path)
    loaded_scores = loaded.score_items(user_index=1, item_indices=candidates)

    np.testing.assert_allclose(original_scores, loaded_scores)
