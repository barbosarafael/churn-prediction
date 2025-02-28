# Import libraries 

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from utils import clean_all_data
import mlflow
import mlflow.sklearn

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

# d) Models

# Defina os parâmetros para cada modelo
param_grids = [
    {
        'classifier': [LogisticRegression()],
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__max_iter': [100, 200, 500]
    },
    {
        'classifier': [DecisionTreeClassifier()],
        'classifier__max_depth': [None, 10, 20, 30, 50],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', None]
    },
    {
        'classifier': [XGBClassifier(use_label_encoder=False, eval_metric = 'logloss')],
        'classifier__n_estimators': [50, 100, 200, 500],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0],
        'classifier__gamma': [0, 0.1, 0.2],
        'classifier__reg_alpha': [0, 0.1, 1],
        'classifier__reg_lambda': [0, 0.1, 1]
    }
]

mlflow.set_tracking_uri('../mlruns')  # Altere para o caminho desejado
mlflow.set_experiment("Churn Prediction Experiment v3")


# Execute o Grid Search e registre no MLflow

def run_and_log_combination(pipeline, params, X_train, y_train, X_test, y_test):
    with mlflow.start_run():
        # Defina os parâmetros no pipeline
        pipeline.set_params(**params)

        # Treine o modelo
        pipeline.fit(X_train, y_train)

        # Avalie o modelo no conjunto de treino e teste
        train_accuracy = pipeline.score(X_train, y_train)
        test_accuracy = pipeline.score(X_test, y_test)

        # Log dos parâmetros e métricas
        mlflow.log_params(params)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Salve o modelo
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"Parâmetros: {params}")
        print(f"Acurácia no treino: {train_accuracy}")
        print(f"Acurácia no teste: {test_accuracy}")

# Execute e registre todas as combinações de parâmetros
for param_grid in param_grids:
    model_name = param_grid['classifier'][0].__class__.__name__
    print(f"\nExecutando combinações para o modelo: {model_name}")

    # Crie um gerador de combinações de parâmetros
    combinations = ParameterGrid(param_grid)

    # Execute e registre cada combinação
    for params in combinations:
        run_and_log_combination(pipeline, params, X_train, y_train, X_test, y_test)



# for param_grid in param_grids:
#     with mlflow.start_run():
#         # Defina o modelo atual
#         model = param_grid['classifier'][0]
#         mlflow.log_param("model", model.__class__.__name__)

#         # Execute o Grid Search
#         grid_search = RandomizedSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
#         grid_search.fit(X, y)

#         # Log dos parâmetros e métricas
#         mlflow.log_params(grid_search.best_params_)
#         mlflow.log_metric("best_score", grid_search.best_score_)

#         # Salve o modelo
#         mlflow.sklearn.log_model(grid_search.best_estimator_, "model")

#         print(f"Melhores parâmetros para {model.__class__.__name__}: {grid_search.best_params_}")
#         print(f"Melhor score: {grid_search.best_score_}")