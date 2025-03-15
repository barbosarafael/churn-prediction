# Import libraries 

from utils_clean_data import load_and_clean_data
from utils_modelling import *
import yaml
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import pandas as pd
import mlflow

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
CALIBRATED_PATH = config['modelling']['calibrated_model_path']

# Load data

X_test = pd.read_csv(PROCESSED_DATA_PATH / 'X_test.csv')
y_test = pd.read_csv(PROCESSED_DATA_PATH / 'y_test.csv')

# Load model 

id_model = '7ab7bfd0c8d54b87a3ce7d6eb2ede953'
best_model_path = f'{MODEL_PATH}/best_model_{id_model}/'

best_model = mlflow.sklearn.load_model(best_model_path)

# Aplicar o pré-processamento aos dados de validação

X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)
print("Pré-processamento aplicado aos dados de validação.")
print(X_test_transformed)

# Calibrar o modelo

best_model_class = best_model.named_steps['classifier']

calibrated_model = CalibratedClassifierCV(best_model_class, method='sigmoid', cv='prefit')
calibrated_model.fit(X_test_transformed, y_test)

# Prever probabilidades calibradas

y_pred_proba_calibrated = calibrated_model.predict_proba(X_test_transformed)[:, 1]

# Avaliar a calibragem com a Curva de Confiabilidade
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba_calibrated, n_bins=10)

# Plotar a Curva de Confiabilidade
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibrado")
plt.plot([0, 1], [0, 1], "k:", label="Perfeitamente Calibrado")
plt.ylabel("Fração de Positivos")
plt.xlabel("Probabilidade Média Prevista")
plt.legend()
plt.title("Curva de Confiabilidade")
plt.savefig(f"confusion_matrix_calibrated.png")
plt.close()
print("Curva de confiabilidade salva como 'confusion_matrix_calibrated.png'.")

# Calcular o Brier Score
brier_score = brier_score_loss(y_test, y_pred_proba_calibrated)
print(f"Brier Score após calibração: {brier_score}")

# Salvar o modelo calibrado em uma nova pasta 
# Melhorar o nome da pasta 