"""Exceções específicas do domínio do projeto."""

from __future__ import annotations


class EcommerceRecommenderError(Exception):
    """Exceção base para todos os erros do projeto."""


class UnsupportedDatasetError(EcommerceRecommenderError):
    """Lançada quando um dataset solicitado não possui carregador registrado."""


class DatasetPathNotFoundError(EcommerceRecommenderError):
    """Lançada quando o caminho de um dataset não existe."""


class UnsupportedModelError(EcommerceRecommenderError):
    """Lançada quando um modelo de recomendação solicitado não é reconhecido."""
