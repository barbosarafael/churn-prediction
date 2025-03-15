# Import libraries 

from utils_clean_data import load_and_clean_data
from utils_modelling import *
import yaml
from pathlib import Path
from sklearn.calibration import calibration_curve
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

# Load data

X_test = pd.read_csv(PROCESSED_DATA_PATH / 'X_test.csv')
y_test = pd.read_csv(PROCESSED_DATA_PATH / 'y_test.csv')

# Load model 

model_dirs = [d for d in Path(MODEL_PATH).iterdir() if d.is_dir() and d.name.startswith("best_model_")]

# model = mlflow.sklearn.load_model()
print(model_dirs)

# Load pipeline 


