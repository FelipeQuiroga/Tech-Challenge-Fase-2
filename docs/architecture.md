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

## Próximas etapas

- Implementar o processamento real em `RetailRocketDataLoader`.
- Pipelines de pré-processamento (interim → processed).
- Modelo neural de recomendação, treinamento, MLflow, DVC e Docker.
