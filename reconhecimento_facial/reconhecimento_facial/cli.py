import csv
from pathlib import Path
from typing import Annotated

import cv2
from loguru import logger
import typer

from reconhecimento_facial.config import (
    AUTORIZADOS_DIR,
    DETECTOR,
    MODELO,
    RESULTADOS_DIR,
    TESTES_DIR,
)
from reconhecimento_facial.dataset import carregar_imagem, listar_imagens, salvar_imagem
from reconhecimento_facial.features import limiar_padrao
from reconhecimento_facial.modeling.predict import reconhecer
from reconhecimento_facial.modeling.train import BaseAutorizados, carregar_base
from reconhecimento_facial.plots import desenhar_resultados

app = typer.Typer(help="Controle de acesso por reconhecimento facial (OpenCV + DeepFace).")

OpcaoModelo = typer.Option(MODELO, help="Modelo de embeddings do DeepFace.")
OpcaoDetector = typer.Option(DETECTOR, help="Detector de rostos do DeepFace.")
OpcaoLimiar = typer.Option(
    None, help="Distância máxima para liberar o acesso. Padrão: limiar do DeepFace para o modelo."
)


def _preparar_base(modelo: str, detector: str, recalcular: bool = False) -> BaseAutorizados:
    base = carregar_base(AUTORIZADOS_DIR, modelo, detector, recalcular=recalcular)
    if not base.embeddings:
        logger.error(f"Nenhum rosto autorizado cadastrado. Adicione fotos em {AUTORIZADOS_DIR}")
        raise typer.Exit(code=1)
    return base


@app.command()
def cadastrar(
    modelo: str = OpcaoModelo,
    detector: str = OpcaoDetector,
    recalcular: bool = typer.Option(False, help="Ignora o cache e recalcula os embeddings."),
) -> None:
    """Gera a base de embeddings a partir das fotos de pessoas autorizadas."""
    base = _preparar_base(modelo, detector, recalcular)
    typer.echo(f"Pessoas autorizadas: {', '.join(base.pessoas)}")


@app.command()
def testar(
    pasta: Annotated[Path, typer.Option(help="Pasta com as fotos de teste.")] = TESTES_DIR,
    saida: Annotated[
        Path, typer.Option(help="Pasta onde as fotos anotadas são salvas.")
    ] = RESULTADOS_DIR,
    modelo: str = OpcaoModelo,
    detector: str = OpcaoDetector,
    limiar: float | None = OpcaoLimiar,
    mostrar: bool = typer.Option(False, help="Exibe cada resultado numa janela do OpenCV."),
) -> None:
    """Verifica cada rosto das fotos de teste e salva as imagens com as caixas de acesso."""
    imagens = listar_imagens(pasta)
    if not imagens:
        logger.error(f"Nenhuma imagem encontrada em {pasta}")
        raise typer.Exit(code=1)

    base = _preparar_base(modelo, detector)
    limiar = limiar if limiar is not None else limiar_padrao(modelo)
    logger.info(f"Modelo {modelo}, detector {detector}, limiar de distância {limiar:.3f}")

    linhas = []
    for caminho in imagens:
        imagem = carregar_imagem(caminho)
        resultados = reconhecer(imagem, base, limiar)
        destino = saida / caminho.relative_to(pasta)
        anotada = desenhar_resultados(imagem, resultados)
        salvar_imagem(anotada, destino)

        if not resultados:
            logger.warning(f"{caminho.name}: nenhum rosto detectado")
            linhas.append([caminho.name, "", "SEM ROSTO", "", ""])
        for i, resultado in enumerate(resultados, start=1):
            status = "LIBERADO" if resultado.autorizado else "NEGADO"
            nome = resultado.nome or "desconhecido"
            mensagem = (
                f"{caminho.name} rosto {i}: {status} ({nome}, distância {resultado.distancia:.3f})"
            )
            (logger.success if resultado.autorizado else logger.warning)(mensagem)
            linhas.append(
                [caminho.name, i, status, resultado.nome or "", f"{resultado.distancia:.4f}"]
            )

        if mostrar:
            cv2.imshow("Reconhecimento facial - tecla para continuar, ESC para sair", anotada)
            if cv2.waitKey(0) == 27:
                break

    if mostrar:
        cv2.destroyAllWindows()

    resumo = saida / "resumo.csv"
    with resumo.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["imagem", "rosto", "acesso", "pessoa", "distancia"])
        escritor.writerows(linhas)
    logger.success(f"Imagens anotadas e resumo salvos em {saida}")


@app.command()
def webcam(
    camera: int = typer.Option(0, help="Índice da câmera."),
    modelo: str = OpcaoModelo,
    detector: str = OpcaoDetector,
    limiar: float | None = OpcaoLimiar,
    intervalo: int = typer.Option(
        5, min=1, help="Reconhece a cada N quadros (o DeepFace é lento demais para todos)."
    ),
) -> None:
    """Reconhecimento em tempo real pela webcam. Pressione Q ou ESC para sair."""
    base = _preparar_base(modelo, detector)
    limiar = limiar if limiar is not None else limiar_padrao(modelo)

    captura = cv2.VideoCapture(camera)
    if not captura.isOpened():
        logger.error(f"Não foi possível abrir a câmera {camera}")
        raise typer.Exit(code=1)

    resultados = []
    quadro_atual = 0
    try:
        while True:
            ok, quadro = captura.read()
            if not ok:
                break
            # Entre um reconhecimento e outro, repete as últimas caixas encontradas
            if quadro_atual % intervalo == 0:
                resultados = reconhecer(quadro, base, limiar)
            quadro_atual += 1

            cv2.imshow(
                "Reconhecimento facial - Q para sair", desenhar_resultados(quadro, resultados)
            )
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        captura.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app()
