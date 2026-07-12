"""Testes dos scripts de estágio ``train`` e ``evaluate`` (dataset sintético)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scripts.evaluate import run as run_evaluate
from scripts.train import run as run_train

_N_USERS = 8
_N_ITEMS = 8


def _write_processed_fixture(processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    def _events(n_rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "user_index": rng.integers(0, _N_USERS, size=n_rows),
                "item_index": rng.integers(0, _N_ITEMS, size=n_rows),
            }
        )

    _events(60).to_parquet(processed_dir / "train.parquet", index=False)
    _events(10).to_parquet(processed_dir / "val.parquet", index=False)
    _events(10).to_parquet(processed_dir / "test.parquet", index=False)
    user_map = {str(i): i for i in range(_N_USERS)}
    item_map = {str(i): i for i in range(_N_ITEMS)}
    (processed_dir / "user_id_map.json").write_text(json.dumps(user_map))
    (processed_dir / "item_id_map.json").write_text(json.dumps(item_map))


def test_train_run_saves_ncf_and_item_knn_artifacts(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    _write_processed_fixture(processed_dir)
    ncf_params = {
        "embedding_dim": 4,
        "hidden_dims": (8,),
        "negatives_per_positive": 2,
        "batch_size": 16,
        "max_epochs": 1,
        "patience": 1,
        "learning_rate": 1e-3,
    }

    summary = run_train(processed_dir, models_dir, seed=42, ncf_params=ncf_params)

    assert summary == {"n_users": _N_USERS, "n_items": _N_ITEMS, "epochs_trained": 1}
    assert (models_dir / "ncf.pt").exists()
    assert (models_dir / "item_knn.joblib").exists()
    assert (models_dir / "train_history.json").exists()


def test_evaluate_run_reports_all_four_metrics_per_model(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    _write_processed_fixture(processed_dir)
    ncf_params = {
        "embedding_dim": 4,
        "hidden_dims": (8,),
        "negatives_per_positive": 2,
        "batch_size": 16,
        "max_epochs": 1,
        "patience": 1,
        "learning_rate": 1e-3,
    }
    run_train(processed_dir, models_dir, seed=42, ncf_params=ncf_params)

    results = run_evaluate(
        processed_dir, models_dir, k=3, n_negatives=4, sample_size=5, seed=42
    )

    assert set(results) == {"ncf", "item_knn"}
    for metrics in results.values():
        assert set(metrics) == {"precision", "recall", "ndcg", "hit_rate"}
