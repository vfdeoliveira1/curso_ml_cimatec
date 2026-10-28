from dataclasses import dataclass, field
from pathlib import Path
import pickle

from loguru import logger
import numpy as np
from tqdm import tqdm

from reconhecimento_facial.config import AUTORIZADOS_DIR, DETECTOR, EMBEDDINGS_PATH, MODELO
from reconhecimento_facial.dataset import carregar_imagem, listar_imagens, nome_da_pessoa
from reconhecimento_facial.features import extrair_faces


@dataclass
class BaseAutorizados:
    """Embeddings das fotos autorizadas; `nomes[i]` é a pessoa de `embeddings[i]`."""

    modelo: str
    detector: str
    assinatura: dict[str, int]
    nomes: list[str] = field(default_factory=list)
    embeddings: list[np.ndarray] = field(default_factory=list)

    @property
    def pessoas(self) -> list[str]:
        return sorted(set(self.nomes))


def assinatura_pasta(pasta: Path) -> dict[str, int]:
    """Arquivos e datas de modificação, usados para saber se o cache está desatualizado."""
    return {
        caminho.relative_to(pasta).as_posix(): caminho.stat().st_mtime_ns
        for caminho in listar_imagens(pasta)
    }


def cadastrar_autorizados(
    pasta: Path = AUTORIZADOS_DIR, modelo: str = MODELO, detector: str = DETECTOR
) -> BaseAutorizados:
    """Calcula o embedding do rosto de cada foto da pasta de pessoas autorizadas."""
    base = BaseAutorizados(modelo=modelo, detector=detector, assinatura=assinatura_pasta(pasta))
    for caminho in tqdm(listar_imagens(pasta), desc="Cadastrando autorizados"):
        faces = extrair_faces(carregar_imagem(caminho), modelo, detector)
        if not faces:
            logger.warning(f"Nenhum rosto encontrado em {caminho.name}; foto ignorada.")
            continue
        if len(faces) > 1:
            logger.warning(f"{len(faces)} rostos em {caminho.name}; usando o maior.")
        face = max(faces, key=lambda f: f.area)
        base.nomes.append(nome_da_pessoa(caminho, pasta))
        base.embeddings.append(face.embedding)
    return base


def salvar_base(base: BaseAutorizados, caminho: Path = EMBEDDINGS_PATH) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("wb") as arquivo:
        pickle.dump(base, arquivo)


def carregar_base(
    pasta: Path = AUTORIZADOS_DIR,
    modelo: str = MODELO,
    detector: str = DETECTOR,
    caminho: Path = EMBEDDINGS_PATH,
    recalcular: bool = False,
) -> BaseAutorizados:
    """Usa os embeddings salvos se ainda correspondem às fotos; senão, recadastra."""
    if not recalcular and caminho.exists():
        with caminho.open("rb") as arquivo:
            base = pickle.load(arquivo)
        if (base.modelo, base.detector, base.assinatura) == (
            modelo,
            detector,
            assinatura_pasta(pasta),
        ):
            return base
        logger.info("Fotos ou configuração mudaram desde o último cadastro; recalculando.")

    base = cadastrar_autorizados(pasta, modelo, detector)
    salvar_base(base, caminho)
    logger.success(
        f"{len(base.embeddings)} fotos de {len(base.pessoas)} pessoas cadastradas em {caminho}"
    )
    return base
