# Import libraries 

import pandas as pd
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, recall_score, accuracy_score, precision_score
from sklearn.impute import SimpleImputer
from utils_clean_data import clean_all_data
from utils_modelling import *
import mlflow
import mlflow.sklearn
from category_encoders import *


# Read data

df = pd.read_csv('../data/processed/train.csv')

# Transform data 

df = clean_all_data(df = df)

# Select the features

features = ['gender', 
            'Partner', 
            'Dependents', 
            'tenure',
            'InternetService', 
            'OnlineSecurity', 
            'OnlineBackup', 
            'DeviceProtection', 
            'TechSupport', 
            'StreamingTV', 
            'StreamingMovies', 
            'Contract', 
            'PaymentMethod', 
            'MonthlyCharges',
            'TotalCharges', 
            'QtdProducts']

# Create X and y

X = df[features]
y = df['Churn']

# Split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Save the data

X_train.to_csv('../data/processed/X_train.csv')
X_test.to_csv('../data/processed/X_test.csv')
y_train.to_csv('../data/processed/y_train.csv')
y_test.to_csv('../data/processed/y_test.csv')

# Modelling 

# Defina os encoders para variáveis numéricas e categóricas
numerical_encoders = {
    'standard_scaler': StandardScaler(),
    'minmax_scaler': MinMaxScaler(),
    'robust_scaler': RobustScaler()
}

categorical_encoders = {
    # Encoders da category_encoders (priorizados)
    'target': TargetEncoder(),  # Codificação baseada na média da variável target
    'count': CountEncoder(),  # Codificação baseada na frequência das categorias
    'binary': BinaryEncoder(),  # Codificação binária (similar ao one-hot, mas mais compacta)
    'basen': BaseNEncoder(base=3),  # Codificação em base N (por padrão, base 2 = binária)
    'catboost': CatBoostEncoder(),  # Codificação inspirada no CatBoost (usa target)
    'helmert': HelmertEncoder(),  # Codificação de Helmert (contrastes)
    'sum': SumEncoder(),  # Codificação de soma (contrastes)
    'polynomial': PolynomialEncoder(),  # Codificação polinomial (contrastes)
    'backward_difference': BackwardDifferenceEncoder(),  # Codificação de diferença reversa (contrastes)
    'james_stein': JamesSteinEncoder(),  # Codificação de James-Stein (usa target)
    'woe': WOEEncoder(),  # Codificação Weight of Evidence (usado em problemas de classificação)
    'leave_one_out': LeaveOneOutEncoder(),  # Codificação Leave-One-Out (usa target, excluindo a própria linha)
    'm_estimate': MEstimateEncoder(),  # Codificação M-Estimate (suavização com target)

    # Encoders do scikit-learn (usados como fallback)
    'onehot': OneHotEncoder(handle_unknown='ignore'),  # Codificação one-hot
    'ordinal': OrdinalEncoder()  # Codificação ordinal
}

# a) Numerical features

numerical_features = X_train.head(1).select_dtypes(include = 'number').columns.tolist()

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# b) Categorical features

categorical_features = X_train.head(1).select_dtypes(include = 'object').columns.tolist()

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# c) Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Crie o pipeline

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())  # Modelo padrão (será substituído no Grid Search)
])

# d) Calculate class weights

class_weights = calculate_class_weights(y_train)

# d) Models

# Defina os parâmetros para cada modelo
param_grids = [
    {
        'classifier': [LogisticRegression(max_iter = 500)],
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__max_iter': [100, 200, 500], 
        'classifier__class_weight': ['balanced', None, class_weights, {0: 1.0, 1: 2.0}, {0: 1.0, 1: 3.0}, {0: 1.0, 1: 4.0}],
        'preprocessor__num': list(numerical_encoders.values()),  # Teste diferentes encoders numéricos
        'preprocessor__cat': list(categorical_encoders.values())  # Teste diferentes encoders categóricos
    },
    {
        'classifier': [DecisionTreeClassifier()],
        'classifier__max_depth': [None, 10, 20, 30, 50],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None],
        'classifier__class_weight': ['balanced', None, class_weights, {0: 1.0, 1: 2.0}, {0: 1.0, 1: 3.0}, {0: 1.0, 1: 4.0}],
        'preprocessor__num': list(numerical_encoders.values()),  # Teste diferentes encoders numéricos
        'preprocessor__cat': list(categorical_encoders.values())  # Teste diferentes encoders categóricos
    },
    {
        'classifier': [XGBClassifier(eval_metric = 'auc')],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0],
        'classifier__gamma': [0, 0.1, 0.2],
        'classifier__reg_alpha': [0, 0.1, 1],
        'classifier__reg_lambda': [0, 0.1, 1], 
        'classifier__scale_pos_weight': [1, 2, 3, 4, class_weights[1]],
        'preprocessor__num': list(numerical_encoders.values()),  # Teste diferentes encoders numéricos
        'preprocessor__cat': list(categorical_encoders.values())  # Teste diferentes encoders categóricos
    }
]

mlflow.set_tracking_uri('../mlruns')  # Altere para o caminho desejado
mlflow.set_experiment("Churn Prediction Experiment v10")

n_iter = 5

# Execute o Random Search e registre no MLflow
def run_and_log_all_combinations(pipeline, param_grid, X_train, y_train, X_test, y_test, n_iter=n_iter):
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

# Execute o Random Search para cada modelo e registre todas as combinações no MLflow
for param_grid in param_grids:
    run_and_log_all_combinations(pipeline, param_grid, X_train, y_train, X_test, y_test, n_iter=n_iter)