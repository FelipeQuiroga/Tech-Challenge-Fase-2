"""Testes das configurações da aplicação."""

import pytest
from pydantic import ValidationError

from ecommerce_recommender.config import Settings, get_settings


def test_settings_load_defaults() -> None:
    settings = get_settings()
    assert settings.project_name == "ecommerce-recommender"
    assert settings.environment == "development"
    assert settings.seed == 42


def test_settings_override_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROJECT_NAME", "custom-name")
    monkeypatch.setenv("SEED", "123")
    monkeypatch.setenv("ENVIRONMENT", "production")
    settings = Settings()
    assert settings.project_name == "custom-name"
    assert settings.seed == 123
    assert settings.environment == "production"


def test_settings_invalid_seed_raises() -> None:
    with pytest.raises(ValidationError):
        Settings(seed=-1)
