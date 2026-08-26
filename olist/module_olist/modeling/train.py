import pandas as pd
from module_olist.modeling.pipeline import (
    create_gradient_boosting_pipeline,
    create_xgboost_pipeline,
    create_lightgbm_pipeline,
)

def train_models(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Treina os modelos Gradient Boosting, XGBoost e LightGBM usando os dados de treino.
    """
    models = {
        "Gradient Boosting": create_gradient_boosting_pipeline(),
        "XGBoost": create_xgboost_pipeline(),
        "LightGBM": create_lightgbm_pipeline(),
        
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"Treinando o modelo {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"Modelo {name} treinado com sucesso!")

    return trained_models