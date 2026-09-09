import pandas as pd
import shap
from scipy import sparse

def prepare_data_for_shap(pipeline, X):
    
    #recupera o pre-processador
    preprocessor = pipeline.named_steps['preprocessor']
    
    #aplica as transformações
    x_transformed = preprocessor.transform(X)
    
    if sparse.issparse(x_transformed):
        x_transformed = x_transformed.toarray()
        
    feature_names = preprocessor.get_feature_names_out()
    
    X_transformed = pd.DataFrame(
        x_transformed,
        columns=feature_names,
        index=X.index
    )
    
    return X_transformed


def create_explainer(pipeline):
    #recupera o modelo
    model = pipeline.named_steps['model']
    
    #cria o explainer
    explainer = shap.TreeExplainer(model)
    
    return explainer

def calculate_shap_values(explainer, X_transformed):
    shap_values = explainer(X_transformed)
    return shap_values