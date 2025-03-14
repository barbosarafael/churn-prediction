# 0. Import libraries

import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, auc, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow


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