"""Estágio DVC ``train``: treina o modelo NCF e ajusta o baseline item-KNN.

Uso:
    poetry run python scripts/train.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import joblib
import mlflow
import mlflow.pytorch
import pandas as pd
import yaml

from ecommerce_recommender.config import Settings, get_settings
from ecommerce_recommender.models import create_recommender
from ecommerce_recommender.models.ncf import NCFRecommender

_PARAMS_PATH = Path("params.yaml")
_EXPERIMENT_NAME = "ecommerce-recommender"
_REGISTERED_MODEL_NAME = "ncf-recommender"


def _load_vocab_sizes(processed_dir: Path) -> tuple[int, int]:
    """Lê o tamanho do vocabulário global de usuários/itens dos mapeamentos.

    Args:
        processed_dir: Diretório com ``user_id_map.json``/``item_id_map.json``.

    Returns:
        Tupla ``(n_users, n_items)``.
    """
    user_map_text = (processed_dir / "user_id_map.json").read_text(encoding="utf-8")
    item_map_text = (processed_dir / "item_id_map.json").read_text(encoding="utf-8")
    return len(json.loads(user_map_text)), len(json.loads(item_map_text))


def _load_ncf_params() -> dict[str, Any]:
    """Lê os hiperparâmetros do NCF em ``params.yaml``.

    Returns:
        Dicionário de hiperparâmetros, com ``hidden_dims`` já como tupla.
    """
    params = yaml.safe_load(_PARAMS_PATH.read_text(encoding="utf-8"))
    ncf_params = dict(params["train"]["ncf"])
    ncf_params["hidden_dims"] = tuple(ncf_params["hidden_dims"])
    return ncf_params


def run(
    processed_dir: Path, models_dir: Path, seed: int, ncf_params: dict[str, Any]
) -> dict[str, Any]:
    """Treina o modelo NCF e ajusta o baseline item-KNN, salvando ambos.

    Args:
        processed_dir: Diretório com treino/val e mapeamentos de id.
        models_dir: Diretório de saída dos artefatos treinados.
        seed: Semente de reprodutibilidade.
        ncf_params: Hiperparâmetros do modelo NCF.

    Returns:
        Resumo do treino (tamanhos de vocabulário, épocas treinadas).
    """
    n_users, n_items = _load_vocab_sizes(processed_dir)
    train_events = pd.read_parquet(processed_dir / "train.parquet")
    val_events = pd.read_parquet(processed_dir / "val.parquet")

    ncf = cast(
        NCFRecommender,
        create_recommender(
            "ncf", n_users=n_users, n_items=n_items, seed=seed, **ncf_params
        ),
    )
    ncf.fit(train_events, val_events=val_events)

    item_knn = create_recommender("item_knn", n_users=n_users, n_items=n_items)
    item_knn.fit(train_events)

    models_dir.mkdir(parents=True, exist_ok=True)
    ncf.save(models_dir / "ncf.pt")
    joblib.dump(item_knn, models_dir / "item_knn.joblib")
    (models_dir / "train_history.json").write_text(
        json.dumps(ncf.history, indent=2), encoding="utf-8"
    )
    return {"n_users": n_users, "n_items": n_items, "epochs_trained": len(ncf.history)}


def _log_ncf_run(settings: Settings, ncf_params: dict[str, Any], seed: int) -> None:
    """Registra o run de treino do NCF no MLflow e o modelo no Registry.

    Args:
        settings: Configurações da aplicação (diretórios, tracking URI).
        ncf_params: Hiperparâmetros usados no treino.
        seed: Semente de reprodutibilidade.
    """
    history_path = settings.models_dir / "train_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    ncf = NCFRecommender.load(settings.models_dir / "ncf.pt")
    with mlflow.start_run(run_name="train-ncf"):
        mlflow.log_params({**ncf_params, "seed": seed})
        for entry in history:
            mlflow.log_metric("train_loss", entry["train_loss"], step=entry["epoch"])
            if entry["val_loss"] is not None:
                mlflow.log_metric("val_loss", entry["val_loss"], step=entry["epoch"])
        mlflow.log_artifact(str(history_path))
        mlflow.pytorch.log_model(
            ncf.module,
            name="model",
            registered_model_name=_REGISTERED_MODEL_NAME,
            serialization_format="pickle",
        )


def _log_item_knn_run(settings: Settings, n_users: int, n_items: int) -> None:
    """Registra o run de ajuste do baseline item-KNN no MLflow.

    Args:
        settings: Configurações da aplicação (diretórios, tracking URI).
        n_users: Quantidade de usuários no vocabulário.
        n_items: Quantidade de itens no vocabulário.
    """
    with mlflow.start_run(run_name="train-item_knn"):
        mlflow.log_params({"n_users": n_users, "n_items": n_items})
        mlflow.log_artifact(str(settings.models_dir / "item_knn.joblib"))


def main() -> None:
    """Ponto de entrada do estágio ``train`` do pipeline DVC."""
    settings = get_settings()
    ncf_params = _load_ncf_params()
    summary = run(
        settings.processed_data_dir, settings.models_dir, settings.seed, ncf_params
    )
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(_EXPERIMENT_NAME)
    _log_ncf_run(settings, ncf_params, settings.seed)
    _log_item_knn_run(settings, summary["n_users"], summary["n_items"])
    print(f"[train] {summary}")


if __name__ == "__main__":
    main()
