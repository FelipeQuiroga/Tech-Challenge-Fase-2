# Arquitetura

## Padrões de projeto

### Strategy Pattern — carregadores de dataset

`DataLoaderStrategy` (`src/ecommerce_recommender/data_loaders/base.py`) define a
interface comum (`load`) para os carregadores. Cada dataset tem uma estratégia
concreta — por exemplo, `RetailRocketDataLoader` — permitindo trocar a fonte de
dados sem alterar o código cliente.

### Factory Pattern — seleção por nome

`create_data_loader(name, source_path)`
(`src/ecommerce_recommender/data_loaders/factory.py`) instancia a estratégia
correta a partir de um registro de nomes. Datasets não suportados resultam em
`UnsupportedDatasetError`.

## Configuração

`Settings` (`config.py`) usa Pydantic Settings, carregando valores do ambiente
e de `.env`, com validação de tipos e da semente.

## Carregamento e pré-processamento dos dados

`RetailRocketDataLoader.load()` (`data_loaders/retailrocket.py`) lê
`events.csv` e retorna um `DataFrame` com uma linha por evento de navegação
(`view`, `addtocart` ou `transaction`).

O módulo `preprocessing/` aplica, em funções puras e testáveis
independentemente:

- `clean_events` (`cleaning.py`): remove duplicatas e linhas com campos
  obrigatórios ausentes, ordena por `timestamp`.
- `encode_user_item_ids` / `IdEncoder` (`encoding.py`): mapeia
  `visitorid`/`itemid` originais (esparsos) para índices contíguos
  `0..N-1`, necessários para camadas de embedding.
- `split_by_time` (`splitting.py`): divide os eventos em treino/validação/
  teste por corte cronológico (sem embaralhar), evitando vazamento de
  informação futura para o passado.

Essas funções ainda não persistem em `data/interim`/`data/processed` — essa
orquestração (leitura → limpeza → encoding → split → escrita em disco) será
feita pelos scripts de CLI do pipeline DVC (`dvc.yaml`), em uma etapa futura,
que apenas chamam essas funções.

## Próximas etapas

- Pipeline DVC (`dvc.yaml`) com estágios `preprocess → feature_eng → train →
  evaluate`, escrevendo os artefatos em `data/interim`/`data/processed`.
- Modelo neural de recomendação (MLP/embedding) em PyTorch, comparação com
  baseline Scikit-Learn.
- Tracking de experimentos e Model Registry no MLflow.
- Containerização com Docker (multi-stage) e `docker-compose.yml`.
