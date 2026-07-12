"""Estágio DVC ``train``: treina o modelo NCF e ajusta o baseline item-KNN.

Uso:
    poetry run python scripts/train.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd
import yaml

from ecommerce_recommender.config import get_settings
from ecommerce_recommender.models import create_recommender
from ecommerce_recommender.models.ncf import NCFRecommender

_PARAMS_PATH = Path("params.yaml")


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


def main() -> None:
    """Ponto de entrada do estágio ``train`` do pipeline DVC."""
    settings = get_settings()
    ncf_params = _load_ncf_params()
    summary = run(
        settings.processed_data_dir, settings.models_dir, settings.seed, ncf_params
    )
    print(f"[train] {summary}")


if __name__ == "__main__":
    main()
