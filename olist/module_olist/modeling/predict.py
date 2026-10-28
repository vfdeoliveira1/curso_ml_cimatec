import json
from pathlib import Path

import joblib
import pandas as pd

from module_olist.config import MODELS_DIR
from module_olist.modeling.split import FEATURES

DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.joblib"
DEFAULT_METADATA_PATH = MODELS_DIR / "metadata.json"


def load_model(model_path: Path = DEFAULT_MODEL_PATH):

	model_path = Path(model_path)
	if not model_path.is_file():
		raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")

	return joblib.load(model_path)


def predict(
	data: pd.DataFrame,
	model_path: Path = DEFAULT_MODEL_PATH,
	metadata_path: Path = DEFAULT_METADATA_PATH,
) -> pd.DataFrame:
	"""
	Faz a predição de atraso dos pedidos usando o modelo treinado.

	Retorna uma cópia dos dados com as colunas ``prediction_probability``
	e ``prediction``. A classe prevista usa o threshold salvo nos metadados.
	"""
	if not isinstance(data, pd.DataFrame):
		raise TypeError("data deve ser um pandas.DataFrame")

	missing_features = [feature for feature in FEATURES if feature not in data.columns]
	if missing_features:
		raise ValueError(
			f"Colunas obrigatórias ausentes para predição: {missing_features}"
		)

	model_path = Path(model_path)
	metadata_path = Path(metadata_path)

	if not model_path.is_file():
		raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
	if not metadata_path.is_file():
		raise FileNotFoundError(f"Metadados não encontrados: {metadata_path}")

	model = joblib.load(model_path)
	with metadata_path.open("r", encoding="utf-8") as file:
		metadata = json.load(file)

	try:
		threshold = float(metadata["threshold"])
	except (KeyError, TypeError, ValueError) as error:
		raise ValueError("Os metadados devem conter um threshold numérico") from error

	if not 0 <= threshold <= 1:
		raise ValueError("O threshold deve estar entre 0 e 1")

	probabilities = model.predict_proba(data[FEATURES])[:, 1]
	predictions = (probabilities >= threshold).astype(int)

	result = data.copy()
	result["prediction_probability"] = probabilities
	result["prediction"] = predictions
	return result
