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

### Strategy + Template Method — recomendadores

`RecommenderStrategy` (`src/ecommerce_recommender/models/base.py`) define os
métodos abstratos `fit`/`score_items` e implementa `recommend` como
método-template (ordena os candidatos pela pontuação de `score_items`,
comum a qualquer implementação concreta). `NCFRecommender` (PyTorch) e
`ItemKnnRecommender` (Scikit-Learn) implementam a mesma interface e são
intercambiáveis na avaliação.

### Factory Pattern — seleção de modelo

`create_recommender(name, n_users, n_items, **kwargs)`
(`src/ecommerce_recommender/models/factory.py`) instancia `"ncf"` ou
`"item_knn"` a partir de um registro de nomes, espelhando o mesmo padrão
usado em `data_loaders/factory.py`. Modelos não suportados resultam em
`UnsupportedModelError`.

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

## Modelo neural (NCF) e treino

`NCFRecommender` (`models/ncf.py`) combina embeddings de usuário/item com
uma MLP (`_NCFModule`), treinada por classificação binária (BCE) com
amostragem negativa. Pontos de implementação relevantes:

- **Batching manual, sem `DataLoader`/`TensorDataset`**: a primeira versão
  usava o `DataLoader` padrão do PyTorch e levava ~739s por época no
  dataset completo (2,2M interações → ~11M linhas/época com negativos). O
  gargalo era o overhead de indexação amostra-a-amostra do
  `TensorDataset.__getitem__`. Reescrito para fatiar tensores diretamente
  (`_iterate_batches`, indexação vetorizada), o que reduziu para ~70s/época
  com `batch_size=32768` — a mesma lógica de amostragem negativa, só sem o
  overhead de coleta por amostra.
- **Early stopping** na perda de validação (`patience` em `params.yaml`).
- **Seeds fixadas** (`torch.manual_seed` + `numpy.random.default_rng`) para
  reprodutibilidade da amostragem negativa e da inicialização dos pesos.
- `save`/`load` persistem pesos e hiperparâmetros (incluindo `hidden_dims`)
  em um único checkpoint `.pt`, sem depender de reconstruir a arquitetura
  manualmente na hora de avaliar.

`ItemKnnRecommender` (`models/item_knn.py`) constrói uma matriz esparsa
item-usuário (`scipy.sparse`) a partir do treino e pontua candidatos pela
similaridade de cosseno máxima a um item conhecido do usuário.

## Avaliação

`scripts/evaluate.py` implementa avaliação por amostragem negativa (padrão
comum em recomendação implícita): para cada interação de teste amostrada,
o item verdadeiro compete contra `n_candidate_negatives` itens desconhecidos
do usuário (excluindo itens vistos em treino/val/teste, para não vazar
falsos negativos). As métricas (`evaluation/metrics.py`) são calculadas
sobre esse ranking: `precision@k`, `recall@k`, `ndcg@k`, `hit_rate@k`.

## MLflow: tracking, Registry e promoção

`scripts/train.py` e `scripts/evaluate.py` mantêm a lógica de negócio em
`run()` (pura, testável, sem tocar no MLflow) e isolam toda a instrumentação
em `main()` (`_log_ncf_run`, `_log_item_knn_run`, `_log_to_mlflow`) — os
testes chamam `run()` diretamente, sem precisar de um tracking server.

- `train-ncf`: loga hiperparâmetros + seed, `train_loss`/`val_loss` por
  época (`step=epoch`), o artefato `train_history.json` e registra o
  próprio modelo (`mlflow.pytorch.log_model(ncf.module, ...,
  registered_model_name="ncf-recommender")`) — `ncf.module` é uma property
  pública que expõe o `nn.Module` interno sem quebrar o encapsulamento do
  `NCFRecommender`.
- `train-item_knn`: loga `n_users`/`n_items` e o artefato `item_knn.joblib`.
- `evaluate`: loga as métricas de ambos os modelos (prefixadas por nome,
  ex. `ncf_hit_rate`) e decide a promoção do modelo (`_promote_if_better`):
  só promove `ncf-recommender` para `Production` (via `Staging`) se o
  `hit_rate` do NCF superar o do baseline nessa avaliação, e arquiva
  automaticamente a versão anterior em Production
  (`archive_existing_versions=True`) para nunca haver duas versões
  simultâneas em produção.

Tracking URI padrão: `sqlite:///mlflow.db` (backend local, sem precisar de
`mlflow server` — que tem um bug de import conhecido nesta combinação
Python 3.14 + MLflow 3.x). O SQLite já suporta o Model Registry
completamente; `mlflow ui --backend-store-uri sqlite:///mlflow.db` expõe a
UI para inspeção.

## Pipeline DVC

`dvc.yaml` define quatro estágios, executáveis via `dvc repro`:

1. **`preprocess`**: `scripts/preprocess.py` — `RetailRocketDataLoader.load()`
   + `clean_events()` → `data/interim/events_clean.parquet`.
2. **`feature_eng`**: `scripts/feature_eng.py` — `encode_user_item_ids()` +
   `split_by_time()` (parâmetros `split.val_fraction`/`split.test_fraction`
   em `params.yaml`) → `data/processed/{train,val,test}.parquet` e
   `{user,item}_id_map.json`.
3. **`train`**: `scripts/train.py` — treina `NCFRecommender` (early stopping)
   e ajusta `ItemKnnRecommender` → `models/ncf.pt`, `models/item_knn.joblib`,
   `models/train_history.json`; loga tudo no MLflow e registra o NCF no
   Model Registry.
4. **`evaluate`**: `scripts/evaluate.py` — compara os dois modelos em
   `data/processed/test.parquet` → `metrics.json` (rastreado como `metrics`
   no `dvc.yaml`, `cache: false` para ficar visível/diffável no Git); loga
   as métricas no MLflow e decide a promoção do NCF a Production.

Todos os estágios rodam via `python -m scripts.<nome>` (não
`python scripts/<nome>.py`) — `evaluate.py` importa constantes de
`scripts/train.py`, e o módulo `scripts` só é resolvido corretamente com a
raiz do projeto no `sys.path`, o que `-m` garante a partir do diretório de
trabalho (a raiz, de onde o `dvc repro` roda).

O dataset bruto (`data/raw/retailrocket/`) é rastreado por
`data/raw/retailrocket.dvc` e versionado em um remote local
(`.dvc-storage/`, configurado em `.dvc/config`). Cada estágio declara seus
`deps` (script + módulos de origem + dados/modelos de entrada) e `outs`, de
forma que `dvc repro` só reexecuta o que de fato mudou.

## Próximas etapas

- Model Card com performance, limitações e vieses.
- Containerização com Docker (multi-stage) e `docker-compose.yml`,
  incluindo um serviço de `mlflow server` (ou apontando para o backend
  SQLite via volume compartilhado).
