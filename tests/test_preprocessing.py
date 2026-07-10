"""Testes dos módulos de limpeza, codificação e divisão temporal."""

import pandas as pd
import pytest

from ecommerce_recommender.preprocessing import (
    IdEncoder,
    clean_events,
    encode_user_item_ids,
    split_by_time,
)


def _sample_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [3, 1, 2, 4],
            "visitorid": [10, 10, 20, 20],
            "event": ["view", "view", "addtocart", "transaction"],
            "itemid": [100, 100, 200, 300],
        }
    )


def test_clean_events_drops_duplicates_and_sorts_by_timestamp() -> None:
    events = pd.concat([_sample_events(), _sample_events().iloc[[0]]])

    cleaned = clean_events(events)

    assert len(cleaned) == 4
    assert list(cleaned["timestamp"]) == sorted(cleaned["timestamp"])


def test_clean_events_drops_rows_with_missing_required_fields() -> None:
    events = _sample_events()
    events.loc[0, "itemid"] = None

    cleaned = clean_events(events)

    assert len(cleaned) == 3


def test_id_encoder_maps_to_contiguous_indices() -> None:
    encoder = IdEncoder().fit(pd.Series([300, 100, 300, 200]))

    encoded = encoder.transform(pd.Series([100, 200, 300]))

    assert encoder.size == 3
    assert set(encoded) == {0, 1, 2}


def test_encode_user_item_ids_adds_index_columns() -> None:
    events = _sample_events()

    encoded, user_encoder, item_encoder = encode_user_item_ids(events)

    assert user_encoder.size == 2
    assert item_encoder.size == 3
    assert set(encoded["user_index"]) == {0, 1}
    assert set(encoded["item_index"]) == {0, 1, 2}


def test_split_by_time_preserves_chronological_order() -> None:
    events = pd.DataFrame({"timestamp": range(100), "value": range(100)})

    split = split_by_time(events, val_fraction=0.2, test_fraction=0.1)

    assert len(split.train) == 70
    assert len(split.val) == 20
    assert len(split.test) == 10
    assert split.train["timestamp"].max() < split.val["timestamp"].min()
    assert split.val["timestamp"].max() < split.test["timestamp"].min()


def test_split_by_time_rejects_invalid_fractions() -> None:
    events = pd.DataFrame({"timestamp": range(10)})
    with pytest.raises(ValueError, match="val_fraction"):
        split_by_time(events, val_fraction=0.6, test_fraction=0.5)
