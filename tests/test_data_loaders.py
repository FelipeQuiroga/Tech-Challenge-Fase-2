"""Testes da fábrica e dos carregadores de dataset."""

from pathlib import Path

import pytest

from ecommerce_recommender.data_loaders import (
    RetailRocketDataLoader,
    available_loaders,
    create_data_loader,
)
from ecommerce_recommender.exceptions import (
    DatasetPathNotFoundError,
    UnsupportedDatasetError,
)


def test_factory_creates_retailrocket_loader(tmp_path: Path) -> None:
    loader = create_data_loader("retailrocket", tmp_path)
    assert isinstance(loader, RetailRocketDataLoader)
    assert loader.source_path == tmp_path


def test_factory_is_case_insensitive(tmp_path: Path) -> None:
    loader = create_data_loader("RetailRocket", tmp_path)
    assert isinstance(loader, RetailRocketDataLoader)


def test_available_loaders_lists_retailrocket() -> None:
    assert "retailrocket" in available_loaders()


def test_factory_unsupported_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedDatasetError):
        create_data_loader("unknown", tmp_path)


def test_loader_missing_path_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(DatasetPathNotFoundError):
        create_data_loader("retailrocket", missing)
