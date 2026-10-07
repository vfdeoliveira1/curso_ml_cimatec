from pathlib import Path

import cv2
import numpy as np

from reconhecimento_facial.config import AUTORIZADOS_DIR, EXTENSOES_IMAGEM


def listar_imagens(pasta: Path) -> list[Path]:
    """Lista recursivamente as imagens de uma pasta, em ordem alfabética."""
    if not pasta.exists():
        return []
    return sorted(
        caminho
        for caminho in pasta.rglob("*")
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_IMAGEM
    )


def nome_da_pessoa(caminho: Path, pasta_base: Path = AUTORIZADOS_DIR) -> str:
    """Identidade da foto: a subpasta da pessoa ou, na raiz, o nome do arquivo."""
    relativo = caminho.relative_to(pasta_base)
    return relativo.parts[0] if len(relativo.parts) > 1 else caminho.stem


def carregar_imagem(caminho: Path) -> np.ndarray:
    """Lê uma imagem em BGR. Usa imdecode porque cv2.imread falha com acentos no caminho."""
    imagem = cv2.imdecode(np.fromfile(caminho, dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagem is None:
        raise ValueError(f"Não foi possível ler a imagem: {caminho}")
    return imagem


def salvar_imagem(imagem: np.ndarray, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(caminho.suffix or ".jpg", imagem)
    if not ok:
        raise ValueError(f"Não foi possível salvar a imagem: {caminho}")
    buffer.tofile(caminho)
