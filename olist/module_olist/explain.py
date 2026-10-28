import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

from module_olist.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
)

from module_olist.modeling.predict import (
    DEFAULT_MODEL_PATH,
    load_model,
)
from module_olist.modeling.split import FEATURES

from module_olist.modeling.interpret import (
    prepare_data_for_shap,
    create_explainer,
    calculate_shap_values,
)

DEFAULT_INPUT_PATH = INTERIM_DATA_DIR / "orders_dataset_refined.csv"
DEFAULT_OUTPUT_PATH = FIGURES_DIR / "shap_summary.png"
DEFAULT_LOCAL_OUTPUT_PATH = FIGURES_DIR / "shap_local.png"


def main(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    local_output_path: Path = DEFAULT_LOCAL_OUTPUT_PATH,
    local_index: int = 0,
) -> None:
    """Gera explicacoes SHAP global e local para o dataset informado."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    local_output_path = Path(local_output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset nao encontrado: {input_path}")

    data = pd.read_csv(input_path)
    missing_features = [feature for feature in FEATURES if feature not in data.columns]
    if missing_features:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing_features}")
    if not 0 <= local_index < len(data):
        raise IndexError(
            f"local_index deve estar entre 0 e {len(data) - 1}: {local_index}"
        )

    model = load_model(model_path)
    X_transformed = prepare_data_for_shap(model, data[FEATURES])
    explainer = create_explainer(model)
    shap_values = calculate_shap_values(explainer, X_transformed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"Grafico SHAP salvo em: {output_path}")

    local_output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.plots.waterfall(shap_values[local_index], max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(local_output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(
        f"Explicacao SHAP local da observacao {local_index} salva em: "
        f"{local_output_path}"
    )


if __name__ == "__main__":
    main()

