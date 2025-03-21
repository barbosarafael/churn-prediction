# Import libraries 

from utils_clean_data import load_and_clean_data
from utils_modelling import *
import yaml
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import pandas as pd
import mlflow

# Load configurations

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

def main():

    # Load data

    X_test = pd.read_csv(PROCESSED_DATA_PATH / 'X_test.csv')
    y_test = pd.read_csv(PROCESSED_DATA_PATH / 'y_test.csv')
    y_test = np.ravel(y_test)

    # Load model 

    id_model = 'cdeb67983fb547a398617fe30b3c58ce'
    best_model_path = f'{MODEL_PATH}/best_model_{id_model}/'

    best_model = mlflow.sklearn.load_model(best_model_path)

    # Apply preprocessing to validation data

    X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)

    # Calibrate model

    best_model_class = best_model.named_steps['classifier']

    calibrated_model = CalibratedClassifierCV(best_model_class, method = 'sigmoid')
    calibrated_model.fit(X_test_transformed, y_test)

    # Predict calibrated probabilities

    y_pred_proba_calibrated = calibrated_model.predict_proba(X_test_transformed)[:, 1]
    
    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba_calibrated)

    # Calculate Brier Score
    brier_score = brier_score_loss(y_test, y_pred_proba_calibrated)
    print(f"Brier Score after calibration: {brier_score}")
    
    # Evaluate calibration with the Reliability Curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba_calibrated, n_bins=10)
    
    # Create folder to save files
    
    calibrated_model_path = f'{CALIBRATED_PATH}/calibrated_model_{id_model}'
    
    os.makedirs(calibrated_model_path, exist_ok = True)

    # Save the calibrated model in a new folder 

    mlflow.sklearn.save_model(calibrated_model, calibrated_model_path)
    
    # Plot the Reliability Curve
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibrated")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    plt.ylabel("Fraction of Positives")
    plt.xlabel("Mean Predicted Probability")
    plt.legend()
    plt.title("Reliability Curve")
    plt.savefig(f"{calibrated_model_path}/reliability_curve.png")
    plt.close()
    
    # Save the optimal threshold
    
    metadata = {"optimal_threshold": optimal_threshold}
    with open(f"{calibrated_model_path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    main()
