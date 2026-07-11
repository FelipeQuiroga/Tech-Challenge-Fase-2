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

A orquestração (leitura → limpeza → encoding → split → escrita em disco) fica
nos scripts de CLI (`scripts/preprocess.py`, `scripts/feature_eng.py`), que
apenas chamam essas funções e persistem os resultados em
`data/interim`/`data/processed` — mantendo a lógica de negócio testável
independentemente da orquestração/IO.

## Pipeline DVC

`dvc.yaml` define dois estágios, executáveis via `dvc repro`:

1. **`preprocess`**: `scripts/preprocess.py` — `RetailRocketDataLoader.load()`
   + `clean_events()` → `data/interim/events_clean.parquet`.
2. **`feature_eng`**: `scripts/feature_eng.py` — `encode_user_item_ids()` +
   `split_by_time()` (parâmetros `split.val_fraction`/`split.test_fraction`
   em `params.yaml`) → `data/processed/{train,val,test}.parquet` e
   `{user,item}_id_map.json`.

O dataset bruto (`data/raw/retailrocket/`) é rastreado por
`data/raw/retailrocket.dvc` e versionado em um remote local
(`.dvc-storage/`, configurado em `.dvc/config`). Cada estágio declara seus
`deps` (script + módulos de origem + dados de entrada) e `outs`, de forma
que `dvc repro` só reexecuta o que de fato mudou.

## Próximas etapas

- Estágios `train` e `evaluate` no `dvc.yaml`.
- Modelo neural de recomendação (MLP/embedding) em PyTorch, comparação com
  baseline Scikit-Learn.
- Tracking de experimentos e Model Registry no MLflow.
- Containerização com Docker (multi-stage) e `docker-compose.yml`.
