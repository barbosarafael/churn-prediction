import json
import joblib
import os
import logging
from datetime import datetime
import boto3
from datetime import datetime
import os
import uuid

# Configura logs detalhados
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configura o DynamoDB
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('ChurnPredictions')  # Substitua pelo nome da sua tabela

def load_artifacts():
    """Carrega modelo e pré-processador com logs de progresso."""
    try:
        logger.info("Iniciando carregamento dos artefatos...")
        MODEL_DIR = "/opt/ml/model"
        
        # Verifica se os arquivos existem
        if not os.path.exists(f"{MODEL_DIR}/model.pkl"):
            raise FileNotFoundError(f"Arquivo model.pkl não encontrado em {MODEL_DIR}")
        if not os.path.exists(f"{MODEL_DIR}/preprocessor.pkl"):
            raise FileNotFoundError(f"Arquivo preprocessor.pkl não encontrado em {MODEL_DIR}")
        
        # Carrega os arquivos
        logger.info("Carregando modelo...")
        model = joblib.load(f"{MODEL_DIR}/model.pkl")
        logger.info("Modelo carregado. Carregando pré-processador...")
        preprocessor = joblib.load(f"{MODEL_DIR}/preprocessor.pkl")
        logger.info("Pré-processador carregado!")
        
        return model, preprocessor
        
    except Exception as e:
        logger.error(f"FALHA NO CARREGAMENTO: {str(e)}")
        raise

# Carrega os artefatos UMA VEZ (durante o Cold Start)
try:
    model, preprocessor = load_artifacts()
    logger.info("Artefatos carregados com sucesso!")
except Exception as e:
    logger.error(f"ERRO CRÍTICO: {str(e)}")
    raise

def save_to_dynamodb(input_data, prediction, probability, model_id):
    try:
        item = {
            'prediction_id': str(uuid.uuid4()),  # ID único
            'timestamp': datetime.now().isoformat(),
            'input_data': json.dumps(input_data),  # Salva como string JSON
            'prediction': int(prediction),
            'probability': float(probability),
            'model_id': model_id
        }
        print(f"Salvando no DynamoDB: {item}")  # Antes do table.put_item()
        table.put_item(Item=item)
    except Exception as e:
        logger.error(f"Erro ao salvar no DynamoDB: {str(e)}")
        # Não interrompe o fluxo se falhar

def lambda_handler(event, context):
    """Função principal com logs em cada etapa."""
    try:
        # Log do evento recebido
        logger.info(f"Evento recebido: {json.dumps(event)}")
        start_time = datetime.now()
        
        # 1. Pré-processamento
        try:
            logger.info("Iniciando pré-processamento...")
            input_data = json.loads(event["body"]) if "body" in event else event
            processed_data = preprocessor.transform([input_data])
            logger.info(f"Pré-processamento concluído. Dados: {processed_data}")
        except Exception as e:
            logger.error(f"ERRO NO PRÉ-PROCESSAMENTO: {str(e)}")
            raise
        
        # 2. Predição
        try:
            logger.info("Iniciando predição...")
            prediction = model.predict(processed_data)[0]
            prediction_proba = model.predict_proba(processed_data)[0][1]
            logger.info(f"Predição concluída: {prediction}")
        except Exception as e:
            logger.error(f"ERRO NA PREDIÇÃO: {str(e)}")
            raise
        
        # 3. Resposta
        response = {
            "statusCode": 200,
            "body": json.dumps({
                "prediction": int(prediction),
                "model_id": os.getenv("MODEL_ID", "unknown"),
                "status": "success",
                "probability": float(prediction_proba)
            })
        }
        logger.info(f"Resposta gerada: {response}")
        
        # 3. Salva no DynamoDB
        save_to_dynamodb(
            input_data=input_data,
            prediction=prediction,
            probability=prediction_proba,
            model_id=os.getenv("MODEL_ID", "unknown")
        )        
        
        return response
        
    except Exception as e:
        logger.error(f"ERRO NA LAMBDA: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }