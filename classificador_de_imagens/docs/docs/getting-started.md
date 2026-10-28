# Começando

Requisitos: Python 3.12 e `uv`. Na pasta `classificador_de_imagens`, instale as
dependências com `uv sync --group dev`. Na primeira execução, o Keras baixa o CIFAR-10.

```powershell
uv run python -m classificador_de_imagens train
uv run python -m classificador_de_imagens evaluate
uv run python -m classificador_de_imagens predict .\imagem.jpg
```

O treino salva o modelo em `models/` e o histórico em `reports/`. Consulte o
[README do projeto](../../README.md) para opções de treino, testes e estrutura completa.
