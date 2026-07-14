# Imagem minima e isolada para o servidor de tracking do MLflow.
#
# Pinada em mlflow==3.8.1 — a MESMA versao usada pelo app (pyproject.toml),
# para cliente e servidor falarem o mesmo protocolo/schema. A partir da 3.9,
# o mlflow server adiciona um recurso de "assistant" cujo import quebra em
# Python 3.14 (`ImportError: cannot import name 'Traversable' from
# 'importlib.abc'`); por isso o teto <3.9 no pyproject.toml. Usa Python 3.12
# (nao 3.14) porque o servidor so precisa do pacote `mlflow`, nao das
# dependencias pesadas de treino (torch, etc.) do restante do app.
FROM python:3.12-slim

RUN pip install --no-cache-dir "mlflow==3.8.1"

WORKDIR /mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
