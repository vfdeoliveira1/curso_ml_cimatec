from dataclasses import dataclass

import numpy as np

from reconhecimento_facial.config import DETECTOR, METRICA_DISTANCIA, MODELO


@dataclass
class Face:
    """Rosto detectado: retângulo (x, y, largura, altura) e embedding facial."""

    x: int
    y: int
    w: int
    h: int
    embedding: np.ndarray

    @property
    def area(self) -> int:
        return self.w * self.h


def extrair_faces(
    imagem: np.ndarray,
    modelo: str = MODELO,
    detector: str = DETECTOR,
) -> list[Face]:
    """Detecta os rostos de uma imagem BGR e calcula o embedding de cada um com o DeepFace."""
    # Import tardio: o DeepFace carrega o TensorFlow, o que deixa lenta qualquer importação
    from deepface import DeepFace

    resultados = DeepFace.represent(
        img_path=imagem,
        model_name=modelo,
        detector_backend=detector,
        enforce_detection=False,
        align=True,
    )

    altura, largura = imagem.shape[:2]
    faces = []
    for resultado in resultados:
        area = resultado["facial_area"]
        # Sem rosto detectado, o DeepFace devolve a imagem inteira com confiança 0
        imagem_inteira = area["w"] >= largura and area["h"] >= altura
        if imagem_inteira or resultado.get("face_confidence", 1) == 0:
            continue
        faces.append(
            Face(
                x=int(area["x"]),
                y=int(area["y"]),
                w=int(area["w"]),
                h=int(area["h"]),
                embedding=np.asarray(resultado["embedding"], dtype=np.float32),
            )
        )
    return faces


def distancia_cosseno(a: np.ndarray, b: np.ndarray) -> float:
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def limiar_padrao(modelo: str = MODELO, metrica: str = METRICA_DISTANCIA) -> float:
    """Distância máxima para o DeepFace considerar que dois rostos são da mesma pessoa."""
    from deepface.modules.verification import find_threshold

    return float(find_threshold(modelo, metrica))
