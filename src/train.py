# Import libraries 

from utils_clean_data import load_and_clean_data
from utils_modelling import *
import yaml
from pathlib import Path

# Carregar configurações
with open('parameters.yml', 'r') as f:
    config = yaml.safe_load(f)

DATA_PATH = Path(config['data']['train_path'])
PROCESSED_DATA_PATH = Path(config['data']['processed_path'])
TEST_SIZE = config['data']['test_size']
RANDOM_STATE = config['data']['random_state']
MLFLOW_TRACKING_URI = config['mlflow']['tracking_uri']
EXPERIMENT_NAME = config['mlflow']['experiment_name']
N_ITER = config['modelling']['n_iter']
FEATURES = config['modelling']['features']
TARGET = config['modelling']['target']
METRIC_NAME = config['modelling']['metric_name']
MODEL_PATH = config['modelling']['model_path']

# Função principal
def main():
    
    # Carregar e limpar dados
    df = load_and_clean_data(DATA_PATH)
    
    # Selecionar features e target
    X, y = select_features_and_target(df, FEATURES, TARGET)
    
    # Dividir e salvar dados
    X_train, X_test, y_train, y_test = split_and_save_data(PROCESSED_DATA_PATH, X, y, TEST_SIZE, RANDOM_STATE)
    
    # Configurar MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Criar pipeline e executar Random Search
    numerical_features = X_train.select_dtypes(include='number').columns.tolist()
    categorical_features = X_train.select_dtypes(include='object').columns.tolist()
    preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression())])
    
    class_weights = calculate_class_weights(y_train)
    param_grids = get_models_and_param_grids(class_weights)
    
    for param_grid in param_grids:
        run_and_log_all_combinations(pipeline, param_grid, X_train, y_train, X_test, y_test, N_ITER)
        

    # Save the best model

    load_and_save_best_model(experiment_name = EXPERIMENT_NAME,
                            metric_name = METRIC_NAME, 
                            save_dir = MODEL_PATH)

if __name__ == "__main__":
    main()