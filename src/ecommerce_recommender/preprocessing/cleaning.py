"""Limpeza dos eventos brutos de navegação."""

from __future__ import annotations

import pandas as pd

_REQUIRED_COLUMNS = ("timestamp", "visitorid", "event", "itemid")


def clean_events(events: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas e linhas inválidas dos eventos brutos.

    Args:
        events: DataFrame bruto de eventos, como retornado por
            ``RetailRocketDataLoader.load``.

    Returns:
        DataFrame limpo, ordenado por ``timestamp`` e com índice reiniciado.
    """
    cleaned = events.dropna(subset=list(_REQUIRED_COLUMNS))
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.sort_values("timestamp", kind="stable")
    return cleaned.reset_index(drop=True)
