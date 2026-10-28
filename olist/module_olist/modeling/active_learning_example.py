"""Exemplo de Active Learning com amostragem por incerteza.

Começamos com poucos exemplos rotulados e, a cada rodada, o modelo escolhe
do "pool" sem rótulo os exemplos em que está mais incerto (menor probabilidade
da classe prevista). Um oráculo (aqui simulado pelo próprio y) revela os
rótulos desses exemplos, que entram no treino. Comparamos com a escolha
aleatória, repetindo o experimento com várias sementes e tirando a média.

Como rodar (dentro da pasta olist):
    uv run python -m module_olist.modeling.active_learning_example
"""

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from module_olist.config import FIGURES_DIR

SEEDS = range(5)
N_INICIAL = 20  # exemplos rotulados no início
N_POR_RODADA = 10  # exemplos rotulados pelo oráculo a cada rodada
N_RODADAS = 20


def criar_modelo():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def escolher_incertos(modelo, X_pool, n):
    """Índices do pool em que o modelo tem menos confiança na classe prevista."""
    confianca = modelo.predict_proba(X_pool).max(axis=1)
    return np.argsort(confianca)[:n]


def escolher_aleatorios(_modelo, X_pool, n, rng):
    return rng.choice(len(X_pool), size=n, replace=False)


def rodar(estrategia, seed, X_train, y_train, X_test, y_test):
    rng = np.random.default_rng(seed)
    rotulados = rng.choice(len(X_train), size=N_INICIAL, replace=False)
    pool = np.setdiff1d(np.arange(len(X_train)), rotulados)

    historico = []
    for _ in range(N_RODADAS + 1):
        modelo = criar_modelo().fit(X_train[rotulados], y_train[rotulados])
        acc = accuracy_score(y_test, modelo.predict(X_test))
        historico.append(acc)

        if estrategia == "incerteza":
            escolhidos = escolher_incertos(modelo, X_train[pool], N_POR_RODADA)
        else:
            escolhidos = escolher_aleatorios(modelo, X_train[pool], N_POR_RODADA, rng)

        # O oráculo revela os rótulos: movemos os exemplos do pool para o treino
        rotulados = np.concatenate([rotulados, pool[escolhidos]])
        pool = np.delete(pool, escolhidos)

    return historico


def main():
    X, y = load_digits(return_X_y=True)
    n_rotulos = [N_INICIAL + i * N_POR_RODADA for i in range(N_RODADAS + 1)]

    resultados = {}
    for estrategia in ("incerteza", "aleatória"):
        execucoes = []
        for seed in SEEDS:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=seed
            )
            execucoes.append(rodar(estrategia, seed, X_train, y_train, X_test, y_test))
        resultados[estrategia] = np.mean(execucoes, axis=0)

    logger.info("Acurácia média no teste (incerteza x aleatória):")
    for n, acc_inc, acc_ale in zip(n_rotulos, *resultados.values()):
        logger.info(f"  {n:3d} rótulos -> {acc_inc:.3f} x {acc_ale:.3f}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    caminho = FIGURES_DIR / "active_learning.png"
    plt.figure(figsize=(8, 5))
    for estrategia, acuracias in resultados.items():
        plt.plot(n_rotulos, acuracias, marker="o", label=estrategia)
    plt.xlabel("Exemplos rotulados")
    plt.ylabel("Acurácia média no teste")
    plt.title("Active Learning: incerteza x aleatória")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=120)
    logger.success(f"Figura salva em {caminho}")


if __name__ == "__main__":
    main()