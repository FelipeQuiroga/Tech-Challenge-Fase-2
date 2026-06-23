"""Definição da estratégia base para carregadores de dataset (Strategy Pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ecommerce_recommender.exceptions import DatasetPathNotFoundError


class DataLoaderStrategy(ABC):
    """Interface comum para todos os carregadores de dataset.

    Cada implementação concreta encapsula a lógica de leitura de um dataset
    específico, permitindo trocar a estratégia sem alterar o código cliente.
    """

    def __init__(self, source_path: Path) -> None:
        """Inicializa o carregador.

        Args:
            source_path: Diretório onde os arquivos brutos do dataset residem.

        Raises:
            DatasetPathNotFoundError: Se ``source_path`` não existir.
        """
        if not source_path.exists():
            raise DatasetPathNotFoundError(
                f"Caminho do dataset não encontrado: {source_path}"
            )
        self._source_path = source_path

    @property
    def source_path(self) -> Path:
        """Retorna o diretório de origem do dataset.

        Returns:
            Caminho configurado para o dataset.
        """
        return self._source_path

    @abstractmethod
    def load(self) -> object:
        """Carrega o dataset a partir do diretório de origem.

        Returns:
            Os dados carregados em uma estrutura adequada à implementação.
        """
        raise NotImplementedError
