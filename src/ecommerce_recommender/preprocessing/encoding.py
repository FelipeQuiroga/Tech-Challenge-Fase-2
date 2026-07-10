"""Codificação de identificadores de usuários e itens em índices contíguos."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class IdEncoder:
    """Mapeia identificadores originais para índices contíguos (``0..N-1``).

    Necessário para camadas de embedding, que exigem índices densos e
    sequenciais em vez dos ids originais e esparsos do dataset.
    """

    _id_to_index: dict[int, int] = field(default_factory=dict)

    def fit(self, ids: pd.Series) -> IdEncoder:
        """Aprende o mapeamento a partir dos ids únicos observados.

        Args:
            ids: Série com os identificadores originais.

        Returns:
            A própria instância, para permitir encadeamento.
        """
        unique_ids = pd.unique(ids)
        self._id_to_index = {raw_id: index for index, raw_id in enumerate(unique_ids)}
        return self

    def transform(self, ids: pd.Series) -> pd.Series:
        """Converte identificadores originais em índices contíguos.

        Args:
            ids: Série com os identificadores originais, todos já
                observados em ``fit``.

        Returns:
            Série de índices inteiros correspondentes.
        """
        return ids.map(self._id_to_index).astype("int32")

    @property
    def size(self) -> int:
        """Retorna a quantidade de identificadores conhecidos.

        Returns:
            Número de ids distintos aprendidos por ``fit``.
        """
        return len(self._id_to_index)


def encode_user_item_ids(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, IdEncoder, IdEncoder]:
    """Adiciona colunas de índices contíguos para usuários e itens.

    Args:
        events: DataFrame de eventos já limpo, com colunas ``visitorid`` e
            ``itemid``.

    Returns:
        Tupla com o DataFrame enriquecido (novas colunas ``user_index`` e
        ``item_index``) e os codificadores ajustados de usuário e item.
    """
    user_encoder = IdEncoder().fit(events["visitorid"])
    item_encoder = IdEncoder().fit(events["itemid"])
    encoded = events.copy()
    encoded["user_index"] = user_encoder.transform(encoded["visitorid"])
    encoded["item_index"] = item_encoder.transform(encoded["itemid"])
    return encoded, user_encoder, item_encoder
