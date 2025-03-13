# 0. Import libraries

from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score, auc, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

def load_kaggle_credentials():
    
    # 1. Load environment variables
    
    """
    Carrega as credenciais do Kaggle da variável de ambiente local.
    
    Essa função carrega as credenciais do Kaggle armazenadas em um arquivo .env
    na pasta raiz do projeto. As credenciais s o necess rias para acessar a API do Kaggle.
    """
    
    load_dotenv()
    
    # 2. Define credentials
    
    os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")
    

def clean_total_charges(df: pd.DataFrame, col_total_charges = 'TotalCharges') -> pd.DataFrame:
    
    df[col_total_charges] = np.where(df[col_total_charges] == " ", 0, df[col_total_charges])
    df[col_total_charges] = df[col_total_charges].astype(float)
    
    return df

def exclude_customer_id(df: pd.DataFrame, col_customer_id = 'customerID') -> pd.DataFrame:
    
    df = df.drop(col_customer_id, axis = 1)
    
    return df

def calculate_qtd_products(df: pd.DataFrame, service_cols: list = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']) -> pd.DataFrame:
    
    df['QtdProducts'] = (df[service_cols] == 'Yes').sum(axis = 1)
    
    return df

def mapping_churn_col(df: pd.DataFrame, col_churn = 'Churn') -> pd.DataFrame:
    
    df[col_churn] = df[col_churn].map({'Yes': 1, 'No': 0})
    
    return df

def clean_all_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df = clean_total_charges(df = df)
    df = exclude_customer_id(df = df)
    df = calculate_qtd_products(df = df)
    df = mapping_churn_col(df = df)
    
    return df    


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