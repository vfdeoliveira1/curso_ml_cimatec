# Classificador de imagens CIFAR-10

Projeto de classificação de imagens baseado no notebook `notebooks/Classificador_Images.ipynb`.
O modelo usa TensorFlow/Keras e uma rede convolucional com três blocos Conv2D/MaxPooling2D.

## Requisitos

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

Instale as dependências e crie o ambiente virtual:

```powershell
uv sync --group dev
```

No Windows, ative o ambiente em PowerShell com:

```powershell
.\.venv\Scripts\Activate.ps1
```

Na primeira execução, o Keras baixa o CIFAR-10 automaticamente.

## Uso

Execute os comandos a partir da pasta `classificador_de_imagens`:

```powershell
python -m classificador_de_imagens --help
python -m classificador_de_imagens train --epochs 10 --batch-size 64
python -m classificador_de_imagens evaluate
python -m classificador_de_imagens predict .\caminho\para\imagem.jpg
```

`predict` redimensiona a imagem informada para 32 × 32 pixels RGB. A imagem deve representar
uma das dez classes do CIFAR-10.

## Artefatos gerados

- `models/cifar10_cnn.keras`: modelo treinado.
- `reports/training_history.json`: métricas por época.
- `reports/metrics.json`: perda e acurácia no conjunto de teste.
- `reports/figures/training_history.png`: curvas de treino e validação.
- `reports/figures/predictions.png`: exemplos de predições no conjunto de teste.

## Testes e estilo

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Estrutura

```text
classificador_de_imagens/
├── classificador_de_imagens/
│   ├── config.py
│   ├── dataset.py
│   ├── features.py
│   ├── plots.py
│   ├── cli.py
│   └── modeling/
│       ├── train.py
│       └── predict.py
├── data/                 # Dados locais (não versionados)
├── docs/                 # Documentação MkDocs
├── models/               # Modelo Keras gerado
├── notebooks/            # Notebook original
├── reports/              # Métricas e gráficos
└── tests/
```
