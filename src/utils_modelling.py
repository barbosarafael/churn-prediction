# 0. Import libraries

import os
import json
import numpy as np
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from category_encoders import *
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import auc, roc_auc_score, confusion_matrix, f1_score, recall_score, accuracy_score, precision_score, roc_curve, precision_recall_curve
from sklearn.impute import SimpleImputer
from datetime import datetime


def calculate_class_weights(y_train):
    """
    Calculates class weights based on label distribution.

    Parameters:
    y_train (array-like): Training set labels.

    Returns:
    dict: Dictionary where keys are class indices and values are the corresponding weights.
    """
    # Calculate class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',  # Balancing strategy
        classes=np.unique(y_train),  # Unique classes in the dataset
        y=y_train  # Training set labels
    )

    # Convert weights to a dictionary
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    return class_weight_dict

def calculate_ks_score(y_true, y_pred_proba):
    """
    Calculates the KS score.

    Parameters:
    y_true (array-like): True labels.
    y_pred_proba (array-like): Predicted probabilities of the positive class.

    Returns:
    float: KS score.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    ks_score = np.max(tpr - fpr)
    return ks_score

def calculate_auc_pr(y_true, y_pred_proba):
    """
    Calculates the AUC-PR (Area Under the Precision-Recall Curve).

    Parameters:
    y_true (array-like): True labels.
    y_pred_proba (array-like): Predicted probabilities of the positive class.

    Returns:
    float: AUC-PR.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auc_pr = auc(recall, precision)
    return auc_pr

def plot_and_log_confusion_matrix(y_true, y_pred, run_name="confusion_matrix"):
    """
    Generates and saves the confusion matrix as an image in MLflow.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    temp_file = "confusion_matrix.png"
    plt.savefig(temp_file)
    plt.close()
    mlflow.log_artifact(temp_file, artifact_path=run_name)
    
def find_optimal_threshold(y_true, y_pred_proba):
    
    """
    Finds the optimal threshold for the model based on the maximum F1 score.

    Parameters:
    y_true (array-like): True labels.
    y_pred_proba (array-like): Predicted probabilities of the positive class.

    Returns:
    float: Optimal threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    return optimal_threshold    

# Select features and target
def select_features_and_target(df, features, target):
    """
    Selects features and target from a given dataframe.

    Parameters:
    df (pd.DataFrame): Input dataframe.
    features (list): List of feature names.
    target (str): Target variable name.

    Returns:
    X (pd.DataFrame): Features dataframe.
    y (pd.Series): Target variable series.
    """
    
    X = df[features]
    y = df[target]
    return X, y

def split_and_save_data(data_path, X, y, test_size=0.2, random_state=42):
    
    """
    Splits the data into training and testing sets and saves the sets to CSV files in the given data path.

    Parameters:
    data_path (str): Path to the directory where the CSV files will be saved.
    X (pd.DataFrame): Features dataframe.
    y (pd.Series): Target variable series.
    test_size (float): Proportion of the dataset to include in the test set. Defaults to 0.2.
    random_state (int): Seed used to shuffle the data. Defaults to 42.

    Returns:
    X_train (pd.DataFrame): Training features dataframe.
    X_test (pd.DataFrame): Testing features dataframe.
    y_train (pd.Series): Training target variable series.
    y_test (pd.Series): Testing target variable series.
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    X_train.to_csv(f'{data_path}/X_train.csv', index=False)
    X_test.to_csv(f'{data_path}/X_test.csv', index=False)
    y_train.to_csv(f'{data_path}/y_train.csv', index=False)
    y_test.to_csv(f'{data_path}/y_test.csv', index=False)
    return X_train, X_test, y_train, y_test

# Define encoders
def get_encoders():
    
    """
    Returns dictionaries of numerical and categorical encoders for data preprocessing.

    The numerical encoders include:
        - 'standard_scaler': StandardScaler
        - 'minmax_scaler': MinMaxScaler
        - 'robust_scaler': RobustScaler

    The categorical encoders include:
        - 'target': TargetEncoder
        - 'count': CountEncoder
        - 'binary': BinaryEncoder
        - 'basen': BaseNEncoder (base=3)
        - 'catboost': CatBoostEncoder
        - 'helmert': HelmertEncoder
        - 'sum': SumEncoder
        - 'polynomial': PolynomialEncoder
        - 'backward_difference': BackwardDifferenceEncoder
        - 'james_stein': JamesSteinEncoder
        - 'woe': WOEEncoder
        - 'leave_one_out': LeaveOneOutEncoder
        - 'm_estimate': MEstimateEncoder
        - 'onehot': OneHotEncoder with handle_unknown='ignore'
        - 'ordinal': OrdinalEncoder

    Returns:
        tuple: A tuple containing two dictionaries: numerical_encoders and categorical_encoders.
    """

    numerical_encoders = {
        'standard_scaler': StandardScaler(),
        'minmax_scaler': MinMaxScaler(),
        'robust_scaler': RobustScaler()
    }

    categorical_encoders = {
        'target': TargetEncoder(),
        'count': CountEncoder(),
        'binary': BinaryEncoder(),
        'basen': BaseNEncoder(base=3),
        'catboost': CatBoostEncoder(),
        'helmert': HelmertEncoder(),
        'sum': SumEncoder(),
        'polynomial': PolynomialEncoder(),
        'backward_difference': BackwardDifferenceEncoder(),
        'james_stein': JamesSteinEncoder(),
        'woe': WOEEncoder(),
        'leave_one_out': LeaveOneOutEncoder(),
        'm_estimate': MEstimateEncoder(),
        'onehot': OneHotEncoder(handle_unknown='ignore'),
        'ordinal': OrdinalEncoder()
    }

    return numerical_encoders, categorical_encoders

# Create preprocessing pipeline
def create_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Creates a preprocessing pipeline.

    The preprocessing pipeline consists of two transformers: a numerical transformer and a categorical transformer.
    The numerical transformer imputes missing values using the median, and scales the data using the StandardScaler.
    The categorical transformer imputes missing values with the string 'unknown', and one-hot encodes the data.
    The transformers are combined using the ColumnTransformer.

    Parameters
    ----------
    numerical_features : list
        A list of numerical feature names
    categorical_features : list
        A list of categorical feature names

    Returns
    -------
    ColumnTransformer
        The preprocessing pipeline
    """
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor


def get_models_and_param_grids(class_weights):
    """
    Returns a list of dictionaries containing the models and their hyperparameters for tuning.
    
    Parameters
    ----------
    class_weights : dict
        A dictionary with the class weights for the classifier.
    
    Returns
    -------
    list
        A list of dictionaries containing the models and their hyperparameters for tuning.
    """
    
    return [
        {
            'classifier': [LogisticRegression(max_iter=1000)],
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__max_iter': [100, 200, 500],
            'classifier__class_weight': ['balanced', None, class_weights, {0: 1.0, 1: 2.0}, {0: 1.0, 1: 3.0}, {0: 1.0, 1: 4.0}],
            'preprocessor__num': list(get_encoders()[0].values()),
            'preprocessor__cat': list(get_encoders()[1].values())
        },
        {
            'classifier': [DecisionTreeClassifier()],
            'classifier__max_depth': [None, 10, 20, 30, 50],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2', None],
            'classifier__class_weight': ['balanced', None, class_weights, {0: 1.0, 1: 2.0}, {0: 1.0, 1: 3.0}, {0: 1.0, 1: 4.0}],
            'preprocessor__num': list(get_encoders()[0].values()),
            'preprocessor__cat': list(get_encoders()[1].values())
        },
        {
            'classifier': [XGBClassifier(eval_metric='auc')],
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7],
            'classifier__subsample': [0.8, 1.0],
            'classifier__colsample_bytree': [0.8, 1.0],
            'classifier__gamma': [0, 0.1, 0.2],
            'classifier__reg_alpha': [0, 0.1, 1],
            'classifier__reg_lambda': [0, 0.1, 1],
            'classifier__scale_pos_weight': [1, 2, 3, 4, class_weights[1]],
            'preprocessor__num': list(get_encoders()[0].values()),
            'preprocessor__cat': list(get_encoders()[1].values())
        }
    ]
    
def run_and_log_all_combinations(pipeline, param_grid, X_train, y_train, X_test, y_test, n_iter):
    """
    Runs a random search for a given pipeline and parameter grid, and logs the parameters and metrics in MLflow.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to use for the random search
    param_grid : dict
        The parameter grid to use for the random search
    X_train : array-like
        The training data
    y_train : array-like
        The target values for the training data
    X_test : array-like
        The testing data
    y_test : array-like
        The target values for the testing data
    n_iter : int
        The number of iterations for the random search

    Returns
    -------
    None
    """
    
    model_name = param_grid['classifier'][0].__class__.__name__
    print(f"\nRunning Random Search for model: {model_name}")

    # Generate parameter combinations using ParameterSampler
    param_sampler = ParameterSampler(param_grid, n_iter=n_iter, random_state=42)

    # Iterate over each parameter combination
    for i, params in enumerate(param_sampler):
        with mlflow.start_run():
            print(f"Combination {i + 1}/{n_iter} for {model_name}")

            # Set parameters in the pipeline
            pipeline.set_params(**params)

            # Train the model
            pipeline.fit(X_train, y_train)

            # Evaluate the model on the training and testing set
            y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            y_test_pred = pipeline.predict(X_test)

            # Calculate metrics
            test_auc = roc_auc_score(y_test, y_test_pred_proba)
            test_recall = recall_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            test_ks = calculate_ks_score(y_test, y_test_pred_proba)
            test_auc_pr = calculate_auc_pr(y_test, y_test_pred_proba)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            cm = confusion_matrix(y_test, y_test_pred)
            tn, fp, fn, tp = cm.ravel()
            
            plot_and_log_confusion_matrix(y_test, y_test_pred)

            # Log parameters and metrics in MLflow
            mlflow.log_params(params)
            mlflow.log_metric("test_auc", test_auc)
            mlflow.log_metric("test_recall", test_recall)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("test_f1", test_f1)
            mlflow.log_metric("test_ks", test_ks)
            mlflow.log_metric("test_auc_pr", test_auc_pr)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("true_negatives", tn)
            mlflow.log_metric("false_positives", fp)
            mlflow.log_metric("false_negatives", fn)
            mlflow.log_metric("true_positives", tp)

            # Save the model (optional)
            mlflow.sklearn.log_model(pipeline, "model")
            
def load_and_save_best_model(experiment_name, metric_name, save_dir):
    
    """
    Loads the best model based on the given metric from the given experiment and saves it to disk.
    
    Parameters
    ----------
    experiment_name : str
        The name of the experiment from which to load the best model
    metric_name : str
        The name of the metric to use for selecting the best model
    save_dir : str
        The directory to which to save the best model
    
    Returns
    -------
    best_model : Pipeline
        The best model
    best_run : Run
        The Run object for the best model
    """
    
    os.makedirs(save_dir, exist_ok=True)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(experiment.experiment_id, order_by=[f"metrics.{metric_name} DESC"])
    best_run = runs[0]
    best_model = mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/model")
    
    print(f"\nBest model: {best_model.named_steps['classifier']}")
    print(f"\nEncoders for numerical variables: {best_model.named_steps['preprocessor'].named_transformers_['num']}")
    print(f"\nEncoders for categorical variables: {best_model.named_steps['preprocessor'].named_transformers_['cat'].__class__.__name__}")
    print(f"\nMetrics of the best run: {best_run.data.metrics}\n")
    
    # Save the model to disk
    model_path = os.path.join(save_dir, f"best_model_{best_run.info.run_id}")
    mlflow.sklearn.save_model(best_model, model_path)

    # Save execution metadata
    metadata = {
        "run_id": best_run.info.run_id,
        "experiment_name": experiment_name,
        "metric_name": metric_name,
        "metric_value": best_run.data.metrics[metric_name],
        "parameters": best_run.data.params,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    metadata_path = os.path.join(model_path, f"metadata_{best_run.info.run_id}.json")

    with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
    return best_model, best_run


def save_model_by_id(run_id, save_dir):
    
    """
    Salva um modelo em um diret rio local com base em seu run_id.

    Parameters
    ----------
    run_id : str
        O run_id do modelo a ser salvo
    save_dir : str
        O diret rio onde o modelo ser  salvo

    Returns
    -------
    model : object
        O objeto do modelo salvo
    """
    
    if not isinstance(run_id, str):
        raise TypeError("run_id deve ser uma string")
    
    if not isinstance(save_dir, str):
        raise TypeError("save_dir deve ser uma string")
    
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        client = MlflowClient()
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        model_path = os.path.join(save_dir, f"best_model_{run_id}")
        mlflow.sklearn.save_model(model, model_path)
        
        print(f'Modelo salvo em: {model_path}')
        
    except mlflow.MlflowException as e:
        if "MODEL NOT FOUND" in str(e):
            print("Modelo não encontrado")
        else:
            raise
    
    return model