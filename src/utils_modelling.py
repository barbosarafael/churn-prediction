# 0. Import libraries

import os
import json
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, auc, roc_curve, precision_recall_curve
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
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, recall_score, accuracy_score, precision_score
from sklearn.impute import SimpleImputer
from datetime import datetime


def calculate_class_weights(y_treino):
    """
    Calcula os pesos das classes com base na distribuição dos rótulos.

    Parâmetros:
    y_treino (array-like): Rótulos do conjunto de treino.

    Retorna:
    dict: Dicionário onde as chaves são os índices das classes e os valores são os pesos correspondentes.
    """
    # Calcula os pesos das classes
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',  # Estratégia de balanceamento
        classes=np.unique(y_treino),  # Classes únicas no conjunto de dados
        y=y_treino  # Rótulos do conjunto de treino
    )

    # Converte os pesos para um dicionário
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    return class_weight_dict

def calcular_ks_score(y_true, y_pred_proba):
    """
    Calcula o KS score.

    Parâmetros:
    y_true (array-like): Rótulos verdadeiros.
    y_pred_proba (array-like): Probabilidades previstas da classe positiva.

    Retorna:
    float: KS score.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    ks_score = np.max(tpr - fpr)
    return ks_score

def calcular_auc_pr(y_true, y_pred_proba):
    """
    Calcula a AUC-PR (Area Under the Precision-Recall Curve).

    Parâmetros:
    y_true (array-like): Rótulos verdadeiros.
    y_pred_proba (array-like): Probabilidades previstas da classe positiva.

    Retorna:
    float: AUC-PR.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auc_pr = auc(recall, precision)
    return auc_pr

def plot_and_log_confusion_matrix(y_true, y_pred, run_name="confusion_matrix"):
    """
    Gera e salva a matriz de confusão como uma imagem no MLflow.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    temp_file = "confusion_matrix.png"
    plt.savefig(temp_file)
    plt.close()
    mlflow.log_artifact(temp_file, artifact_path=run_name)
    
def find_optimal_threshold(y_true, y_pred_proba):
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    return optimal_threshold    

# Select features and target
def select_features_and_target(df, features, target):
    X = df[features]
    y = df[target]
    return X, y

def split_and_save_data(data_path, X, y, test_size=0.2, random_state=42):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    X_train.to_csv(f'{data_path}/X_train.csv', index=False)
    X_test.to_csv(f'{data_path}/X_test.csv', index=False)
    y_train.to_csv(f'{data_path}/y_train.csv', index=False)
    y_test.to_csv(f'{data_path}/y_test.csv', index=False)
    return X_train, X_test, y_train, y_test

# Define encoders
def get_encoders():
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
    return [
        {
            'classifier': [LogisticRegression(max_iter=500)],
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
    model_name = param_grid['classifier'][0].__class__.__name__
    print(f"\nExecutando Random Search para o modelo: {model_name}")

    # Gere combinações de parâmetros usando ParameterSampler
    param_sampler = ParameterSampler(param_grid, n_iter=n_iter, random_state=42)

    # Itere sobre cada combinação de parâmetros
    for i, params in enumerate(param_sampler):
        with mlflow.start_run():
            print(f"Combinação {i + 1}/{n_iter} para {model_name}")

            # Defina os parâmetros no pipeline
            pipeline.set_params(**params)

            # Treine o modelo
            pipeline.fit(X_train, y_train)

            # Avalie o modelo no conjunto de treino e teste
            y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            
            optimal_threshold = find_optimal_threshold(y_test, y_test_pred_proba)
            y_test_pred = (y_test_pred_proba >= optimal_threshold).astype(int)

            # Calcule as métricas
            test_auc = roc_auc_score(y_test, y_test_pred_proba)
            test_recall = recall_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            test_ks = calcular_ks_score(y_test, y_test_pred_proba)
            test_auc_pr = calcular_auc_pr(y_test, y_test_pred_proba)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            cm = confusion_matrix(y_test, y_test_pred)
            tn, fp, fn, tp = cm.ravel()
            
            plot_and_log_confusion_matrix(y_test, y_test_pred)

            # Log dos parâmetros e métricas no MLflow
            mlflow.log_params(params)
            mlflow.log_metric("optimal_threshold", optimal_threshold)
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

            # Salve o modelo (opcional)
            mlflow.sklearn.log_model(pipeline, "model")
            
def load_and_save_best_model(experiment_name, metric_name, save_dir):
    
    os.makedirs(save_dir, exist_ok=True)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(experiment.experiment_id, order_by=[f"metrics.{metric_name} DESC"])
    best_run = runs[0]
    best_model = mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/model")
    
    print(f"\nMelhor modelo: {best_model.named_steps['classifier']}")
    print(f"\nEncoders das variáveis numéricas: {best_model.named_steps['preprocessor'].named_transformers_['num']}")
    print(f"\nEncoders das variáveis categóricas: {best_model.named_steps['preprocessor'].named_transformers_['cat'].__class__.__name__}")
    print(f"\nMétricas da melhor run: {best_run.data.metrics}\n")
    
    # Salva o modelo em disco
    # model_path = os.path.join(save_dir, f"best_model_{best_run.info.run_id}")
    # mlflow.sklearn.save_model(best_model, model_path)

    # # Salva metadados da execução
    # metadata = {
    #     "run_id": best_run.info.run_id,
    #     "experiment_name": experiment_name,
    #     "metric_name": metric_name,
    #     "metric_value": best_run.data.metrics[metric_name],
    #     "parameters": best_run.data.params,
    #     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # }
    # metadata_path = os.path.join(model_path, f"metadata_{best_run.info.run_id}.json")

    # with open(metadata_path, "w") as f:
    #         json.dump(metadata, f, indent=4)
            
    return best_model, best_run