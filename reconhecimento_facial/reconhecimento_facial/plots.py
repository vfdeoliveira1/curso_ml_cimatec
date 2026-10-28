import unicodedata

import cv2
import numpy as np

from reconhecimento_facial.config import (
    COR_LIBERADO,
    COR_NEGADO,
    TEXTO_LIBERADO,
    TEXTO_NEGADO,
)
from reconhecimento_facial.modeling.predict import Reconhecimento

FONTE = cv2.FONT_HERSHEY_SIMPLEX


def _sem_acentos(texto: str) -> str:
    # As fontes Hershey do OpenCV só desenham ASCII
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


def _escrever(
    imagem: np.ndarray, texto: str, x: int, y: int, cor: tuple, escala: float, acima: bool
) -> None:
    """Escreve o texto em branco numa faixa da cor da caixa, acima ou abaixo da linha `y`."""
    espessura = max(1, round(escala * 2))
    (largura, altura), base = cv2.getTextSize(texto, FONTE, escala, espessura)
    margem = max(2, round(4 * escala))
    altura_faixa = altura + base + 2 * margem
    # Se não couber fora da caixa (borda da imagem), a faixa vai para dentro dela
    if acima:
        topo = y - altura_faixa if y - altura_faixa >= 0 else y
    else:
        topo = y if y + altura_faixa <= imagem.shape[0] else y - altura_faixa
    cv2.rectangle(imagem, (x, topo), (x + largura + 2 * margem, topo + altura_faixa), cor, -1)
    cv2.putText(
        imagem,
        texto,
        (x + margem, topo + margem + altura),
        FONTE,
        escala,
        (255, 255, 255),
        espessura,
        cv2.LINE_AA,
    )


def desenhar_resultados(imagem: np.ndarray, resultados: list[Reconhecimento]) -> np.ndarray:
    """Desenha uma caixa verde (acesso liberado) ou vermelha (acesso negado) em cada rosto."""
    anotada = imagem.copy()
    # Escala do texto e da borda proporcional ao tamanho da imagem
    escala = max(0.5, min(anotada.shape[:2]) / 900)
    espessura = max(2, round(escala * 3))

    for resultado in resultados:
        face = resultado.face
        cor = COR_LIBERADO if resultado.autorizado else COR_NEGADO
        texto = TEXTO_LIBERADO if resultado.autorizado else TEXTO_NEGADO

        cv2.rectangle(
            anotada, (face.x, face.y), (face.x + face.w, face.y + face.h), cor, espessura
        )
        _escrever(anotada, texto, face.x, face.y, cor, escala, acima=True)
        if resultado.nome:
            nome = _sem_acentos(resultado.nome)
            _escrever(anotada, nome, face.x, face.y + face.h, cor, escala * 0.8, acima=False)
    return anotada
