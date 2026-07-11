"""Estágio DVC ``preprocess``: limpa os eventos brutos do RetailRocket.

Uso:
    poetry run python scripts/preprocess.py
"""

from __future__ import annotations

from pathlib import Path

from ecommerce_recommender.config import get_settings
from ecommerce_recommender.data_loaders import create_data_loader
from ecommerce_recommender.preprocessing import clean_events

_DATASET_NAME = "retailrocket"
_OUTPUT_FILENAME = "events_clean.parquet"


def run(raw_dir: Path, interim_dir: Path) -> Path:
    """Carrega e limpa os eventos brutos, gravando o resultado em parquet.

    Args:
        raw_dir: Diretório com os arquivos brutos do dataset.
        interim_dir: Diretório onde o resultado limpo será gravado.

    Returns:
        Caminho do arquivo parquet gerado.
    """
    loader = create_data_loader(_DATASET_NAME, raw_dir)
    cleaned = clean_events(loader.load())

    interim_dir.mkdir(parents=True, exist_ok=True)
    output_path = interim_dir / _OUTPUT_FILENAME
    cleaned.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    """Ponto de entrada do estágio ``preprocess`` do pipeline DVC."""
    settings = get_settings()
    raw_dir = settings.raw_data_dir / _DATASET_NAME
    output_path = run(raw_dir, settings.interim_data_dir)
    print(f"[preprocess] eventos limpos gravados em {output_path}")


if __name__ == "__main__":
    main()
