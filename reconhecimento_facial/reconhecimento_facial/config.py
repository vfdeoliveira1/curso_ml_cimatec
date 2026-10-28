import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Reduz o log do TensorFlow (usado internamente pelo DeepFace)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
# Fotos das pessoas autorizadas. Use uma subpasta por pessoa (data/autorizados/maria/1.jpg)
# ou um arquivo por pessoa (data/autorizados/maria.jpg); o nome vira a identidade.
AUTORIZADOS_DIR = DATA_DIR / "autorizados"
# Fotos que serão verificadas pelo sistema
TESTES_DIR = DATA_DIR / "testes"

MODELS_DIR = PROJ_ROOT / "models"
EMBEDDINGS_PATH = MODELS_DIR / "embeddings_autorizados.pkl"

REPORTS_DIR = PROJ_ROOT / "reports"
RESULTADOS_DIR = REPORTS_DIR / "resultados"

# Reconhecimento facial
EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MODELO = "Facenet512"
DETECTOR = "opencv"
METRICA_DISTANCIA = "cosine"

# Cores no padrão BGR do OpenCV
COR_LIBERADO = (0, 200, 0)
COR_NEGADO = (0, 0, 255)
TEXTO_LIBERADO = "ACESSO LIBERADO"
TEXTO_NEGADO = "ACESSO NEGADO"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
