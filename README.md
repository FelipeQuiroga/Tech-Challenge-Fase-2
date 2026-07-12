# E-commerce Recommender

Sistema de recomendação de produtos de e-commerce baseado no dataset
[RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

Este repositório cobre, por enquanto, as **Etapas 1 (Clean Code e
Estrutura)**, **2 (Ambiente e Dependências)** e **3 (Containerização e
Versionamento, parte DVC)**: dataset versionado, pipeline reprodutível de
dados e um modelo neural (PyTorch) já treinado e comparado com um baseline.
MLflow, Model Registry e Docker serão adicionados em etapas futuras.

## Objetivo

Construir, de forma incremental, um sistema de recomendação de produtos a
partir do comportamento de usuários (visualizações, carrinho e compras) do
dataset RetailRocket, com código limpo, ambiente reprodutível e qualidade
automatizada.

## Estrutura de pastas

```
.
├── src/ecommerce_recommender/   # Código-fonte do pacote (src layout)
│   ├── config.py                # Configurações (Pydantic Settings)
│   ├── exceptions.py            # Exceções específicas do domínio
│   ├── data_loaders/            # Strategy + Factory de carregadores
│   ├── preprocessing/           # Limpeza, encoding e split temporal
│   ├── models/                  # Strategy + Factory de recomendadores
│   │   ├── ncf.py               # Modelo neural (PyTorch): embeddings + MLP
│   │   └── item_knn.py          # Baseline item-item (Scikit-Learn)
│   └── evaluation/               # Métricas de ranking (precision/recall/…)
├── tests/                       # Testes (pytest)
├── data/
│   ├── raw/                     # Dados brutos (não versionados)
│   ├── interim/                 # Dados intermediários
│   └── processed/               # Dados processados
├── models/                      # Modelos treinados (não versionados no Git)
├── configs/                     # Configurações em YAML (referência)
├── scripts/                     # Scripts utilitários e estágios do DVC
│   ├── validate_env.py          # Validação do ambiente
│   ├── preprocess.py            # Estágio DVC: raw -> interim
│   ├── feature_eng.py           # Estágio DVC: interim -> processed
│   ├── train.py                 # Estágio DVC: treina NCF + ajusta baseline
│   └── evaluate.py              # Estágio DVC: compara modelos (métricas)
├── dvc.yaml / dvc.lock          # Pipeline DVC (estágios e hashes)
├── params.yaml                  # Parâmetros do pipeline (seed, splits, modelo)
├── metrics.json                 # Métricas da última avaliação (versionado)
└── docs/                        # Documentação
```

## Instalação com Poetry

Requer **Python 3.14** e **Poetry ≥ 2.0** (o `pyproject.toml` usa o formato
PEP 621 — `[project] dependencies`, que só é totalmente suportado a partir do
Poetry 2.0; versões anteriores falham em `poetry check`/`poetry install`).

```bash
poetry env use 3.14
poetry install
```

## Configuração do `.env`

Copie o exemplo e ajuste os valores conforme necessário:

```bash
cp .env.example .env
```

Todas as variáveis podem ser sobrescritas por variáveis de ambiente ou pelo
arquivo `.env`. As principais são: `ENVIRONMENT`, `PROJECT_NAME`, `LOG_LEVEL`,
`SEED`, os diretórios de dados/modelos e `MLFLOW_TRACKING_URI`.

## Execução dos testes

```bash
poetry run pytest
```

## Lint e formatação (Ruff)

```bash
poetry run ruff format .
poetry run ruff check .
```

## Pre-commit

```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

## Validação do ambiente

```bash
poetry run python scripts/validate_env.py
```

O script verifica a versão do Python, importa as bibliotecas principais,
carrega as configurações, garante a existência dos diretórios e informa se o
CUDA está disponível.

## Onde colocar o dataset

Baixe o dataset do
[Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
(login necessário, sem necessidade de token de API), extraia os arquivos e
coloque-os em:

```
data/raw/retailrocket/
├── events.csv
├── item_properties_part1.csv
├── item_properties_part2.csv
└── category_tree.csv
```

Os dados brutos **não devem ser adicionados ao Git** (já ignorados no
`.gitignore`); quem clonar o repositório recupera os mesmos bytes via
`dvc pull` (ver seção abaixo), sem precisar baixar do Kaggle de novo.

`RetailRocketDataLoader.load()` lê apenas `events.csv` (2,75M eventos de
navegação — views, adições ao carrinho e transações), que é a fonte de
interações usuário-item usada pelo pipeline de recomendação.

## Pipeline de dados (DVC)

O dataset bruto é versionado com DVC (`data/raw/retailrocket.dvc`) e o
pipeline completo é definido em `dvc.yaml`, com 4 estágios:

```
preprocess (raw → interim) → feature_eng (interim → processed) → train → evaluate
```

- **`preprocess`**: lê `events.csv`, remove duplicatas/linhas inválidas
  (`clean_events`) e grava `data/interim/events_clean.parquet`.
- **`feature_eng`**: codifica `visitorid`/`itemid` em índices contíguos para
  embeddings (`encode_user_item_ids`), divide em treino/validação/teste por
  corte cronológico (`split_by_time`, parâmetros em `params.yaml`) e grava
  `data/processed/{train,val,test}.parquet` + `{user,item}_id_map.json`.
- **`train`**: treina o modelo neural (`NCFRecommender`) com early stopping
  e ajusta o baseline (`ItemKnnRecommender`); salva `models/ncf.pt`,
  `models/item_knn.joblib` e `models/train_history.json`.
- **`evaluate`**: compara os dois modelos em `data/processed/test.parquet`
  por amostragem negativa (ranking) e grava `metrics.json`.

Reproduzir o pipeline do zero:

```bash
poetry run dvc repro
```

Buscar os dados já versionados (sem reprocessar), depois de clonar o
repositório:

```bash
poetry run dvc pull
```

O remote configurado (`local-storage`) é uma pasta local (`.dvc-storage/`,
fora do Git) — suficiente para demonstrar o fluxo `dvc add`/`dvc push`/
`dvc pull` sem depender de credenciais de nuvem. Para uso em equipe real,
troque por um remote S3/Azure/GCS (`dvc remote add -d ... s3://...`).

> **Nota (Windows):** se o caminho do projeto for muito longo (pasta de
> usuário + nome do repositório ultrapassando ~260 caracteres), `dvc repro`/
> `dvc commit` pode falhar com `WinError 3` ao gravar no *run-cache* interno
> do DVC (nomes de arquivo com hash duplicado). Solução: aponte o cache do
> DVC para um caminho curto, só nesta máquina (não versionado):
> `dvc cache dir "C:\dvc-cache\<nome-do-projeto>" --local`.

## Modelo de recomendação

Dois modelos implementam a mesma interface (`RecommenderStrategy`,
`src/ecommerce_recommender/models/base.py` — Strategy + Template Method) e
são instanciados por nome via `create_recommender` (Factory,
`src/ecommerce_recommender/models/factory.py`):

- **`ncf`** — rede neural (PyTorch): embeddings de usuário/item concatenados
  e passados por uma MLP, treinada como classificação binária (interagiu
  vs. amostra negativa) com amostragem negativa por época e early stopping
  na perda de validação. Seeds fixadas (`params.yaml: seed`).
- **`item_knn`** — baseline de filtragem colaborativa item-item: similaridade
  de cosseno (`sklearn.metrics.pairwise.cosine_similarity`) entre os itens
  candidatos e os itens que o usuário já conhece.

Avaliação (estágio `evaluate`, `metrics.json`): para cada interação de teste
amostrada, o item verdadeiro é ranqueado contra `n_candidate_negatives`
itens desconhecidos do usuário; mede-se `precision@k`, `recall@k`,
`ndcg@k` e `hit_rate@k` (parâmetros em `params.yaml: evaluate`).

**Resultado da última execução** (`k=10`, 99 negativos por interação, 2.000
interações de teste amostradas):

| Modelo     | precision@10 | recall@10 | ndcg@10 | hit_rate@10 |
| ---------- | -----------: | --------: | ------: | ----------: |
| `ncf`      |        0.054 |     0.541 |   0.330 |       0.541 |
| `item_knn` |        0.004 |     0.040 |   0.031 |       0.040 |

O NCF supera o baseline em todas as métricas por uma margem grande (~13x em
hit rate) — esperado: ele aprende uma representação latente sobre toda a
distribuição de interações, enquanto o baseline só enxerga similaridade
direta entre os poucos itens que cada usuário já tocou, em um catálogo de
235 mil itens.
