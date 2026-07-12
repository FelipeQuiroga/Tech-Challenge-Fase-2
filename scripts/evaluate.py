"""Estágio DVC ``evaluate``: compara NCF vs. baseline item-KNN por métricas.

Avalia por amostragem negativa: para cada interação de teste amostrada,
ranqueia o item verdadeiro contra ``n_candidate_negatives`` itens que o
usuário não conhece, e mede se/quão bem o modelo o coloca no topo.

Uso:
    poetry run python scripts/evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from ecommerce_recommender.config import get_settings
from ecommerce_recommender.evaluation import evaluate_rankings
from ecommerce_recommender.models.base import RecommenderStrategy
from ecommerce_recommender.models.ncf import NCFRecommender

_PARAMS_PATH = Path("params.yaml")
_METRICS_PATH = Path("metrics.json")


def _load_evaluate_params() -> dict[str, Any]:
    """Lê os parâmetros de avaliação em ``params.yaml``.

    Returns:
        Dicionário com ``k``, ``n_candidate_negatives`` e ``eval_sample_size``.
    """
    params = yaml.safe_load(_PARAMS_PATH.read_text(encoding="utf-8"))
    return dict(params["evaluate"])


def _known_items_by_user(*event_frames: pd.DataFrame) -> dict[int, set[int]]:
    """Combina os itens conhecidos por usuário em treino+val+teste.

    Usado para não sortear como "negativo" um item que o usuário já
    conhece por outro conjunto.

    Args:
        *event_frames: DataFrames com colunas ``user_index``/``item_index``.

    Returns:
        Itens conhecidos por usuário.
    """
    combined = pd.concat([df[["user_index", "item_index"]] for df in event_frames])
    return combined.groupby("user_index")["item_index"].apply(set).to_dict()


def _sample_negative_candidates(
    known_items: set[int], n_items: int, n_negatives: int, rng: np.random.Generator
) -> np.ndarray:
    """Amostra itens desconhecidos do usuário para compor os candidatos.

    Args:
        known_items: Itens já conhecidos pelo usuário (a excluir).
        n_items: Tamanho do vocabulário global de itens.
        n_negatives: Quantidade de candidatos negativos a amostrar.
        rng: Gerador de números aleatórios (reprodutível).

    Returns:
        Array de ids de itens candidatos negativos. Pode ter menos que
        ``n_negatives`` se o usuário já conhece quase todo o catálogo.
    """
    target = min(n_negatives, max(n_items - len(known_items), 0))
    candidates: set[int] = set()
    max_attempts = 50
    for _ in range(max_attempts):
        if len(candidates) >= target:
            break
        batch = rng.integers(0, n_items, size=n_negatives * 2)
        candidates.update(int(item) for item in batch if item not in known_items)
    return np.array(list(candidates)[:target])


def _rank_model(
    model: RecommenderStrategy,
    interactions: pd.DataFrame,
    known_items_by_user: dict[int, set[int]],
    n_items: int,
    n_negatives: int,
    k: int,
    seed: int,
) -> tuple[dict[int, list[int]], dict[int, set[int]]]:
    """Gera recomendações e verdade-fundamental por interação de teste.

    Args:
        model: Estratégia de recomendação já treinada.
        interactions: Interações de teste amostradas para avaliação.
        known_items_by_user: Itens conhecidos por usuário (treino+val+teste).
        n_items: Tamanho do vocabulário global de itens.
        n_negatives: Candidatos negativos por interação.
        k: Tamanho do corte do ranking.
        seed: Semente de reprodutibilidade da amostragem de negativos.

    Returns:
        Tupla ``(recomendações, itens relevantes)``, indexadas pela posição
        da interação na amostra.
    """
    rng = np.random.default_rng(seed)
    recommendations: dict[int, list[int]] = {}
    relevant: dict[int, set[int]] = {}
    for row_id, row in enumerate(interactions.itertuples(index=False)):
        known = known_items_by_user.get(row.user_index, set())
        negatives = _sample_negative_candidates(known, n_items, n_negatives, rng)
        candidates = np.append(negatives, row.item_index)
        recommendations[row_id] = model.recommend(row.user_index, candidates, k)
        relevant[row_id] = {int(row.item_index)}
    return recommendations, relevant


def run(
    processed_dir: Path,
    models_dir: Path,
    k: int,
    n_negatives: int,
    sample_size: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Avalia o NCF e o baseline item-KNN no conjunto de teste.

    Args:
        processed_dir: Diretório com treino/val/teste e mapeamentos de id.
        models_dir: Diretório com os modelos treinados.
        k: Tamanho do corte do ranking.
        n_negatives: Candidatos negativos por interação avaliada.
        sample_size: Quantidade de interações de teste amostradas.
        seed: Semente de reprodutibilidade.

    Returns:
        Métricas de ranking por modelo (``ncf``, ``item_knn``).
    """
    train_events = pd.read_parquet(processed_dir / "train.parquet")
    val_events = pd.read_parquet(processed_dir / "val.parquet")
    test_events = pd.read_parquet(processed_dir / "test.parquet")
    item_map_text = (processed_dir / "item_id_map.json").read_text(encoding="utf-8")
    n_items = len(json.loads(item_map_text))

    sample_n = min(sample_size, len(test_events))
    sample = test_events.sample(n=sample_n, random_state=seed)
    known_items_by_user = _known_items_by_user(train_events, val_events, test_events)

    models: dict[str, RecommenderStrategy] = {
        "ncf": NCFRecommender.load(models_dir / "ncf.pt"),
        "item_knn": joblib.load(models_dir / "item_knn.joblib"),
    }
    results = {}
    for name, model in models.items():
        recommendations, relevant = _rank_model(
            model, sample, known_items_by_user, n_items, n_negatives, k, seed
        )
        results[name] = evaluate_rankings(recommendations, relevant, k)
    return results


def main() -> None:
    """Ponto de entrada do estágio ``evaluate`` do pipeline DVC."""
    settings = get_settings()
    eval_params = _load_evaluate_params()
    results = run(
        settings.processed_data_dir,
        settings.models_dir,
        k=eval_params["k"],
        n_negatives=eval_params["n_candidate_negatives"],
        sample_size=eval_params["eval_sample_size"],
        seed=settings.seed,
    )
    _METRICS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[evaluate] {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
