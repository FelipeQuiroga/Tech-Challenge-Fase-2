"""Script de validação do ambiente de desenvolvimento.

Verifica a versão do Python, importa as bibliotecas principais, carrega as
configurações, garante a existência dos diretórios de dados e informa se o
CUDA está disponível. Retorna código de saída diferente de zero em caso de
falha.

Uso:
    poetry run python scripts/validate_env.py
"""

from __future__ import annotations

import importlib
import sys

REQUIRED_PYTHON = (3, 14)
REQUIRED_LIBRARIES = (
    "torch",
    "pandas",
    "numpy",
    "sklearn",
    "mlflow",
    "dvc",
    "pydantic_settings",
    "dotenv",
)


def check_python_version() -> bool:
    """Verifica se a versão do Python atende ao mínimo exigido.

    Returns:
        ``True`` se a versão for compatível, ``False`` caso contrário.
    """
    current = sys.version_info[:2]
    ok = current >= REQUIRED_PYTHON
    status = "OK" if ok else "FALHA"
    print(
        f"[{status}] Python {current[0]}.{current[1]} "
        f"(mínimo {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]})"
    )
    return ok


def check_libraries() -> bool:
    """Tenta importar todas as bibliotecas principais.

    Returns:
        ``True`` se todas as bibliotecas forem importadas com sucesso.
    """
    all_ok = True
    for library in REQUIRED_LIBRARIES:
        try:
            importlib.import_module(library)
            print(f"[OK] import {library}")
        except ImportError as error:
            all_ok = False
            print(f"[FALHA] import {library}: {error}")
    return all_ok


def check_settings_and_directories() -> bool:
    """Carrega as configurações e garante a existência dos diretórios.

    Returns:
        ``True`` se as configurações carregarem e os diretórios existirem.
    """
    try:
        from ecommerce_recommender.config import get_settings
    except ImportError as error:
        print(f"[FALHA] importar configurações: {error}")
        return False

    settings = get_settings()
    print(f"[OK] configurações carregadas (projeto={settings.project_name})")

    for directory in settings.data_directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"[OK] diretório disponível: {directory}")
    return True


def check_cuda() -> bool:
    """Informa se o CUDA está disponível (não é uma falha se ausente).

    Returns:
        Sempre ``True`` — a ausência de CUDA não invalida o ambiente.
    """
    try:
        import torch

        available = torch.cuda.is_available()
        print(f"[INFO] CUDA disponível: {available}")
    except ImportError:
        print("[INFO] torch indisponível; não foi possível checar CUDA")
    return True


def main() -> int:
    """Executa todas as verificações do ambiente.

    Returns:
        ``0`` se todas as verificações obrigatórias passarem, ``1`` caso falhe.
    """
    print("== Validação do ambiente ==")
    checks = (
        check_python_version(),
        check_libraries(),
        check_settings_and_directories(),
        check_cuda(),
    )
    if all(checks):
        print("\nAmbiente validado com sucesso.")
        return 0
    print("\nFalha na validação do ambiente.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
