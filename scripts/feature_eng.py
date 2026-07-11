"""Estágio DVC ``feature_eng``: codifica ids e divide em treino/val/teste.

Uso:
    poetry run python scripts/feature_eng.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from ecommerce_recommender.config import get_settings
from ecommerce_recommender.preprocessing import encode_user_item_ids, split_by_time

_INPUT_FILENAME = "events_clean.parquet"
_PARAMS_PATH = Path("params.yaml")


def _load_split_fractions() -> tuple[float, float]:
    """Lê as frações de validação/teste de ``params.yaml``.

    Returns:
        Tupla ``(val_fraction, test_fraction)``.
    """
    params = yaml.safe_load(_PARAMS_PATH.read_text(encoding="utf-8"))
    split_params = params["split"]
    return split_params["val_fraction"], split_params["test_fraction"]


def _write_id_mapping(mapping: dict[int, int], output_path: Path) -> None:
    """Grava um mapeamento de id original para índice em JSON.

    Args:
        mapping: Dicionário de id original para índice contíguo.
        output_path: Caminho do arquivo JSON de saída.
    """
    serializable = {str(raw_id): index for raw_id, index in mapping.items()}
    output_path.write_text(json.dumps(serializable), encoding="utf-8")


def run(
    interim_dir: Path,
    processed_dir: Path,
    val_fraction: float,
    test_fraction: float,
) -> None:
    """Codifica ids, divide os eventos e grava os artefatos processados.

    Args:
        interim_dir: Diretório com ``events_clean.parquet``.
        processed_dir: Diretório de saída para treino/val/teste e
            mapeamentos de id.
        val_fraction: Fração dos eventos reservada à validação.
        test_fraction: Fração dos eventos reservada ao teste.
    """
    events = pd.read_parquet(interim_dir / _INPUT_FILENAME)
    encoded, user_encoder, item_encoder = encode_user_item_ids(events)
    split = split_by_time(
        encoded, val_fraction=val_fraction, test_fraction=test_fraction
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    split.train.to_parquet(processed_dir / "train.parquet", index=False)
    split.val.to_parquet(processed_dir / "val.parquet", index=False)
    split.test.to_parquet(processed_dir / "test.parquet", index=False)
    _write_id_mapping(user_encoder.to_mapping(), processed_dir / "user_id_map.json")
    _write_id_mapping(item_encoder.to_mapping(), processed_dir / "item_id_map.json")


def main() -> None:
    """Ponto de entrada do estágio ``feature_eng`` do pipeline DVC."""
    settings = get_settings()
    val_fraction, test_fraction = _load_split_fractions()
    run(
        settings.interim_data_dir,
        settings.processed_data_dir,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    print("[feature_eng] treino/val/teste e mapeamentos de id gravados")


if __name__ == "__main__":
    main()
