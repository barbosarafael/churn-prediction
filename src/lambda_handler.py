import json
import mlflow
import pandas as pd
import joblib
from pathlib import Path
import traceback
import os

# Configurações
MODEL_ID = 'cdeb67983fb547a398617fe30b3c58ce'  # Substitua pelo seu ID real
MODEL_DIR = "/opt/ml/model/"

def load_model():
    try:
        model_path = f"/opt/ml/model/model_{os.environ['MODEL_ID']}.pkl"
        return joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(f"Files in /opt/ml/model: {os.listdir('/opt/ml/model')}")
        raise

def load_preprocessor():
    """Carrega o pré-processador"""
    try:
        preprocessor_path = Path(MODEL_DIR) / f"preprocessor_{MODEL_ID}.pkl"
        return joblib.load(preprocessor_path)
    except Exception as e:
        print(f"Erro ao carregar pré-processador: {str(e)}")
        raise

def preprocess_input(raw_input, preprocessor):
    """Pré-processa os dados de entrada"""
    try:
        # Converte para DataFrame
        input_df = pd.DataFrame([raw_input])
        
        # Verifica colunas faltantes
        required_columns = preprocessor.get_feature_names_out()
        missing_cols = set(required_columns) - set(input_df.columns)
        
        if missing_cols:
            print(f"Colunas faltantes: {missing_cols}")
            # Adiciona colunas faltantes com valores padrão
            for col in missing_cols:
                input_df[col] = 0
        
        return preprocessor.transform(input_df)
        
    except Exception as e:
        print(f"Erro no pré-processamento: {str(e)}")
        raise

def lambda_handler(event, context):
    """Handler principal para o Lambda"""
    try:
        # 1. Carrega artefatos
        model = load_model()
        preprocessor = load_preprocessor()
        
        # 2. Processa input
        raw_input = json.loads(event['body']) if 'body' in event else event
        
        # 3. Pré-processamento
        processed_data = preprocess_input(raw_input, preprocessor)
        
        # 4. Predição
        prediction = model.predict(processed_data)[0]
        result = {
            'prediction': int(prediction),
            'model_id': MODEL_ID,
            'status': 'success'
        }
        
        # 5. Adiciona probabilidades se disponível
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(processed_data)[0]
            result['probability'] = float(proba[1])  # Assume classe 1 é positiva
            
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
        
    except Exception as e:
        error_response = {
            'status': 'error',
            'error': str(e),
            'model_id': MODEL_ID,
            'stack_trace': traceback.format_exc()
        }
        print(f"Erro completo: {json.dumps(error_response, indent=2)}")
        
        return {
            'statusCode': 500,
            'body': json.dumps(error_response)
        }