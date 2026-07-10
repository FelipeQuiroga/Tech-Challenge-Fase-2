"""Divisão temporal dos eventos em treino, validação e teste."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Conjuntos de treino, validação e teste particionados por tempo."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split_by_time(
    events: pd.DataFrame,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> TemporalSplit:
    """Divide os eventos em treino/validação/teste por ordem cronológica.

    Usa os eventos mais recentes (por ``timestamp``) como teste, os
    imediatamente anteriores como validação, e o restante como treino —
    evitando vazamento de informação futura para o passado.

    Args:
        events: DataFrame de eventos limpo, com coluna ``timestamp``.
        val_fraction: Fração dos eventos (após o teste) reservada à
            validação.
        test_fraction: Fração final dos eventos reservada ao teste.

    Returns:
        ``TemporalSplit`` com os três conjuntos particionados.

    Raises:
        ValueError: Se a soma das frações não for menor que 1.
    """
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction deve ser menor que 1")

    ordered = events.sort_values("timestamp", kind="stable").reset_index(drop=True)
    n_rows = len(ordered)
    test_start = int(n_rows * (1 - test_fraction))
    val_start = int(n_rows * (1 - test_fraction - val_fraction))

    train = ordered.iloc[:val_start].reset_index(drop=True)
    val = ordered.iloc[val_start:test_start].reset_index(drop=True)
    test = ordered.iloc[test_start:].reset_index(drop=True)
    return TemporalSplit(train=train, val=val, test=test)
