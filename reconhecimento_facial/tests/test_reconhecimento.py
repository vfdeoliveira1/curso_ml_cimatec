import numpy as np
import pytest

from reconhecimento_facial.config import COR_LIBERADO, COR_NEGADO
from reconhecimento_facial.dataset import (
    carregar_imagem,
    listar_imagens,
    nome_da_pessoa,
    salvar_imagem,
)
from reconhecimento_facial.features import Face, distancia_cosseno
from reconhecimento_facial.modeling.predict import Reconhecimento, identificar
from reconhecimento_facial.modeling.train import BaseAutorizados
from reconhecimento_facial.plots import desenhar_resultados


def _face(embedding, x=40, y=60, w=80, h=80):
    return Face(x=x, y=y, w=w, h=h, embedding=np.asarray(embedding, dtype=np.float32))


@pytest.fixture
def base():
    return BaseAutorizados(
        modelo="Facenet512",
        detector="opencv",
        assinatura={},
        nomes=["maria", "joao"],
        embeddings=[np.array([1.0, 0.0]), np.array([0.0, 1.0])],
    )


def test_nome_da_pessoa_usa_subpasta_ou_nome_do_arquivo(tmp_path):
    assert nome_da_pessoa(tmp_path / "maria" / "foto1.jpg", tmp_path) == "maria"
    assert nome_da_pessoa(tmp_path / "joao.png", tmp_path) == "joao"


def test_salvar_e_carregar_imagem_com_acento_no_caminho(tmp_path):
    imagem = np.full((20, 30, 3), 128, dtype=np.uint8)
    caminho = tmp_path / "joão" / "foto.png"
    salvar_imagem(imagem, caminho)

    assert listar_imagens(tmp_path) == [caminho]
    np.testing.assert_array_equal(carregar_imagem(caminho), imagem)


def test_distancia_cosseno():
    assert distancia_cosseno(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == pytest.approx(0.0)
    assert distancia_cosseno(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_identificar_libera_rosto_proximo_de_um_autorizado(base):
    resultado = identificar(_face([0.1, 1.0]), base, limiar=0.3)
    assert resultado.autorizado
    assert resultado.nome == "joao"


def test_identificar_nega_rosto_desconhecido(base):
    resultado = identificar(_face([1.0, 1.0]), base, limiar=0.1)
    assert not resultado.autorizado
    assert resultado.nome is None


def test_identificar_nega_quando_nao_ha_autorizados():
    vazia = BaseAutorizados(modelo="Facenet512", detector="opencv", assinatura={})
    assert not identificar(_face([1.0, 0.0]), vazia, limiar=0.3).autorizado


@pytest.mark.parametrize("autorizado, cor", [(True, COR_LIBERADO), (False, COR_NEGADO)])
def test_desenhar_resultados_usa_cor_do_acesso(autorizado, cor):
    imagem = np.zeros((300, 300, 3), dtype=np.uint8)
    face = _face([1.0, 0.0])
    resultado = Reconhecimento(face, autorizado=autorizado, nome=None, distancia=0.0)

    anotada = desenhar_resultados(imagem, [resultado])

    # Pixel no meio da borda direita da caixa
    borda = anotada[face.y + face.h // 2, face.x + face.w]
    assert tuple(int(c) for c in borda) == cor
    assert imagem.sum() == 0  # a imagem original não é alterada
