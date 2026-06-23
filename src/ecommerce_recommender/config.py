"""Configurações da aplicação carregadas via Pydantic Settings.

As configurações podem ser sobrescritas por variáveis de ambiente ou por um
arquivo ``.env`` localizado na raiz do projeto.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Configurações centrais do projeto.

    Attributes:
        environment: Ambiente de execução (development, staging ou production).
        project_name: Nome legível do projeto.
        log_level: Nível de log utilizado pela aplicação.
        seed: Semente para reprodutibilidade dos experimentos.
        raw_data_dir: Diretório dos dados brutos.
        interim_data_dir: Diretório dos dados intermediários.
        processed_data_dir: Diretório dos dados processados.
        models_dir: Diretório onde os modelos treinados são salvos.
        mlflow_tracking_uri: Endereço do servidor de tracking do MLflow.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: Literal["development", "staging", "production"] = "development"
    project_name: str = "ecommerce-recommender"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    seed: int = 42

    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    interim_data_dir: Path = PROJECT_ROOT / "data" / "interim"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "models"

    mlflow_tracking_uri: str = Field(default="http://127.0.0.1:5000")

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, value: int) -> int:
        """Garante que a semente seja um inteiro não negativo.

        Args:
            value: Valor da semente informado.

        Returns:
            A semente validada.

        Raises:
            ValueError: Se a semente for negativa.
        """
        if value < 0:
            raise ValueError("seed deve ser um inteiro não negativo")
        return value

    @property
    def data_directories(self) -> tuple[Path, ...]:
        """Retorna todos os diretórios de dados e de modelos.

        Returns:
            Tupla com os diretórios gerenciados pela aplicação.
        """
        return (
            self.raw_data_dir,
            self.interim_data_dir,
            self.processed_data_dir,
            self.models_dir,
        )


def get_settings() -> Settings:
    """Cria e retorna uma instância das configurações.

    Returns:
        Instância de :class:`Settings` carregada do ambiente e do ``.env``.
    """
    return Settings()
