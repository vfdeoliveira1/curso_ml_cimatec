Projeto de experimentação com o dataset público de e-commerce de Brasília.

## Visão geral

Este repositório contém um pipeline de ciência de dados estruturado para carregar, processar e modelar dados do conjunto Olist. O foco está em organizar dados, gerar features, treinar modelos e executar inferência em um projeto Python modular.

## Estrutura do projeto

- `data/`
  - `raw/` — dados originais e imutáveis
  - `interim/` — dados intermediários criados durante o processamento
  - `processed/` — dados finais prontos para modelagem
  - `external/` — fontes externas e dados complementares
- `docs/` — documentação do projeto com MkDocs
- `models/` — modelos treinados e artefatos de machine learning
- `notebooks/` — notebooks Jupyter para exploração e análise
- `reports/` — relatórios gerados e resultados de visualização
- `module_olist/` — código fonte Python do projeto
- `pyproject.toml` — configuração do projeto e dependências

## Módulos principais

- `module_olist/config.py` — caminhos e configurações do projeto
- `module_olist/dataset.py` — transformação e preparação de dados brutos
- `module_olist/features.py` — geração de features para modelagem
- `module_olist/modeling/train.py` — treinamento de modelos
- `module_olist/modeling/predict.py` — inferência e geração de previsões
- `module_olist/plots.py` — visualizações e gráficos

## Requisitos

- Python 3.14
- dependências listadas em `pyproject.toml`

## Instalação

1. Crie e ative um ambiente virtual Python:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instale as dependências:

```powershell
pip install -e .
```

3. Instale dependências extras para desenvolvimento, se necessário:

```powershell
pip install ruff mkdocs pytest
```

## Uso

Os módulos Typer do pacote `module_olist` são executados como scripts Python.

- Processar dados:

```powershell
python -m module_olist.dataset
```

- Gerar features:

```powershell
python -m module_olist.features
```

- Treinar modelo:

```powershell
python -m module_olist.modeling.train
```

- Fazer previsões:

```powershell
python -m module_olist.modeling.predict
```

> Observação: os scripts atuais contêm código de exemplo e mensagens de log. Substitua a lógica de placeholder pelas etapas reais de processamento, modelagem e inferência.

## Desenvolvimento

- Executar testes:

```powershell
python -m pytest
```

- Verificar formatação e lint:

```powershell
ruff check module_olist
```

- Construir documentação:

```powershell
mkdocs build
```

- Servir documentação localmente:

```powershell
mkdocs serve
```

## Notas

- `module_olist/config.py` define os diretórios de dados e relatórios utilizados pelo pipeline.
- Os arquivos em `data/raw/` são considerados a origem dos dados e não devem ser alterados diretamente.
- Este projeto usa `pyproject.toml` como fonte única de configuração de packaging e dependências.

