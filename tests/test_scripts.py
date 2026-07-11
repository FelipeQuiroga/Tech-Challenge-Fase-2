"""Testes dos scripts de estágio do pipeline DVC (preprocess, feature_eng)."""

import json
from pathlib import Path

import pandas as pd
from scripts.feature_eng import run as run_feature_eng
from scripts.preprocess import run as run_preprocess


def test_preprocess_run_writes_clean_events_parquet(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "events.csv").write_text(
        "timestamp,visitorid,event,itemid,transactionid\n"
        "2,10,view,100,\n"
        "1,10,view,100,\n"
        "2,10,view,100,\n"
    )
    interim_dir = tmp_path / "interim"

    output_path = run_preprocess(raw_dir, interim_dir)

    events = pd.read_parquet(output_path)
    assert len(events) == 2
    assert list(events["timestamp"]) == [1, 2]


def test_feature_eng_run_writes_splits_and_id_maps(tmp_path: Path) -> None:
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    events = pd.DataFrame(
        {
            "timestamp": range(20),
            "visitorid": [1, 2] * 10,
            "event": ["view"] * 20,
            "itemid": [100, 200] * 10,
        }
    )
    events.to_parquet(interim_dir / "events_clean.parquet", index=False)
    processed_dir = tmp_path / "processed"

    run_feature_eng(interim_dir, processed_dir, val_fraction=0.2, test_fraction=0.2)

    train = pd.read_parquet(processed_dir / "train.parquet")
    val = pd.read_parquet(processed_dir / "val.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")
    assert len(train) + len(val) + len(test) == 20

    user_map = json.loads((processed_dir / "user_id_map.json").read_text())
    item_map = json.loads((processed_dir / "item_id_map.json").read_text())
    assert set(user_map) == {"1", "2"}
    assert set(item_map) == {"100", "200"}
