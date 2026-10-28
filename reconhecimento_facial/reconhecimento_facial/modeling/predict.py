from dataclasses import dataclass

import numpy as np

from reconhecimento_facial.features import Face, distancia_cosseno, extrair_faces
from reconhecimento_facial.modeling.train import BaseAutorizados


@dataclass
class Reconhecimento:
    """Resultado da verificação de um rosto contra a base de autorizados."""

    face: Face
    autorizado: bool
    nome: str | None
    distancia: float


def identificar(face: Face, base: BaseAutorizados, limiar: float) -> Reconhecimento:
    """Compara o rosto com todas as fotos autorizadas e libera se a mais próxima passar no limiar."""
    if not base.embeddings:
        return Reconhecimento(face, autorizado=False, nome=None, distancia=float("inf"))

    distancias = [distancia_cosseno(face.embedding, e) for e in base.embeddings]
    indice = int(np.argmin(distancias))
    distancia = distancias[indice]
    autorizado = distancia <= limiar
    return Reconhecimento(
        face,
        autorizado=autorizado,
        nome=base.nomes[indice] if autorizado else None,
        distancia=distancia,
    )


def reconhecer(imagem: np.ndarray, base: BaseAutorizados, limiar: float) -> list[Reconhecimento]:
    """Detecta todos os rostos da imagem e decide o acesso de cada um."""
    faces = extrair_faces(imagem, base.modelo, base.detector)
    return [identificar(face, base, limiar) for face in faces]
