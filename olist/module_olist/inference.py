import argparse
from pathlib import Path

from loguru import logger
import pandas as pd

from module_olist.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from module_olist.modeling.predict import (
	DEFAULT_METADATA_PATH,
	DEFAULT_MODEL_PATH,
	predict,
)

DEFAULT_INPUT_PATH = INTERIM_DATA_DIR / "orders_dataset_refined.csv"
DEFAULT_OUTPUT_PATH = PROCESSED_DATA_DIR / "predictions.csv"


def run_inference(
	input_path: Path = DEFAULT_INPUT_PATH,
	output_path: Path = DEFAULT_OUTPUT_PATH,
	model_path: Path = DEFAULT_MODEL_PATH,
	metadata_path: Path = DEFAULT_METADATA_PATH,
) -> pd.DataFrame:
	"""Executa a inferencia em um CSV e salva os resultados."""
	input_path = Path(input_path)
	output_path = Path(output_path)

	if not input_path.is_file():
		raise FileNotFoundError(f"Dataset de entrada nao encontrado: {input_path}")

	data = pd.read_csv(input_path)
	predictions = predict(
		data=data,
		model_path=model_path,
		metadata_path=metadata_path,
	)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	predictions.to_csv(output_path, index=False)
	logger.success(f"Predicoes salvas em: {output_path}")

	return predictions


def main() -> None:
	"""Permite executar a inferencia pela linha de comando."""
	parser = argparse.ArgumentParser(
		description="Executa inferencia com o modelo treinado do projeto."
	)
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
	parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
	parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
	args = parser.parse_args()

	run_inference(
		input_path=args.input,
		output_path=args.output,
		model_path=args.model,
		metadata_path=args.metadata,
	)


if __name__ == "__main__":
	main()
