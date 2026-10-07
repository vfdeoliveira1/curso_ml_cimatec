# reconhecimento_facial

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Controle de acesso por reconhecimento facial com **OpenCV** e **DeepFace**. O sistema compara
cada rosto de uma foto com as fotos de pessoas autorizadas e marca o rosto com uma caixa:

- **verde, "ACESSO LIBERADO"** (com o nome da pessoa), quando o rosto corresponde a alguém autorizado;
- **vermelha, "ACESSO NEGADO"**, quando não corresponde a ninguém.

## Como funciona

1. **Cadastro**: o OpenCV detecta o rosto de cada foto em `data/autorizados` e o modelo
   `Facenet512` do DeepFace gera um vetor de 512 números (embedding) para ele. Os embeddings ficam
   salvos em `models/embeddings_autorizados.pkl` e só são recalculados quando as fotos mudam.
2. **Verificação**: para cada rosto de uma foto de teste, calcula-se a distância de cosseno até
   cada embedding autorizado. Se a menor distância for menor ou igual ao limiar (0,30 para o
   Facenet512, o valor padrão do DeepFace), o acesso é liberado.
3. **Resultado**: o OpenCV desenha as caixas e salva as imagens em `reports/resultados`, junto com
   um `resumo.csv`.

## Instalação

```bash
make create_environment
make requirements        # equivale a: UV_LINK_MODE=copy uv sync
```

`UV_LINK_MODE=copy` é necessário porque o projeto está no OneDrive, que não aceita hardlinks.
Na primeira execução, o DeepFace baixa os pesos do Facenet512 (~95 MB) para `~/.deepface`.

## Uso

1. Coloque as fotos das pessoas autorizadas em `data/autorizados`, de um destes jeitos:
   - uma subpasta por pessoa, com várias fotos (recomendado): `data/autorizados/maria/1.jpg`, `.../maria/2.jpg`
   - um arquivo por pessoa: `data/autorizados/joao.jpg`

   Cada foto deve ter um único rosto, de frente e bem iluminado. Várias fotos por pessoa
   (ângulos e iluminações diferentes) reduzem os acessos negados por engano.
2. Coloque as fotos a verificar em `data/testes`.
3. Execute:

```bash
python -m reconhecimento_facial cadastrar   # gera os embeddings dos autorizados
python -m reconhecimento_facial testar      # processa data/testes -> reports/resultados
python -m reconhecimento_facial webcam      # reconhecimento em tempo real (Q para sair)
```

Ou `make cadastrar`, `make testar` e `make webcam`. Opções úteis (veja `--help` de cada comando):

| Opção | Descrição |
|---|---|
| `--limiar 0.35` | Distância máxima para liberar o acesso: menor é mais rígido, maior é mais permissivo |
| `--detector retinaface` | Troca o detector de rostos do DeepFace (o padrão é `opencv`) |
| `--mostrar` | Em `testar`, exibe cada resultado numa janela |
| `--recalcular` | Em `cadastrar`, ignora o cache de embeddings |

As fotos em `data/` e os resultados ficam fora do Git (`.gitignore`); só a estrutura de pastas é
versionada.

## Organização do projeto

```
├── Makefile                   <- make requirements | cadastrar | testar | webcam | test | lint
├── pyproject.toml             <- Dependências e configuração do ruff
├── data
│   ├── autorizados            <- Fotos das pessoas autorizadas a usar o sistema
│   └── testes                 <- Fotos a serem verificadas
├── models                     <- Embeddings dos autorizados (gerado pelo `cadastrar`)
├── reports
│   └── resultados             <- Fotos com as caixas de acesso + resumo.csv
├── tests                      <- Testes com pytest
└── reconhecimento_facial      <- Código-fonte
    ├── __main__.py            <- Ponto de entrada: python -m reconhecimento_facial
    ├── cli.py                 <- Comandos cadastrar, testar e webcam
    ├── config.py              <- Caminhos, modelo, detector, cores e textos
    ├── dataset.py             <- Leitura e gravação de imagens
    ├── features.py            <- Detecção de rostos e embeddings com o DeepFace
    ├── modeling
    │   ├── train.py           <- Cadastro dos autorizados (base de embeddings)
    │   └── predict.py         <- Decide se cada rosto é liberado ou negado
    └── plots.py               <- Desenha as caixas verde/vermelha com o OpenCV
```

--------
