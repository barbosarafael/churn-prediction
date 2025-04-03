import json
import joblib
import os
import logging
from datetime import datetime

# Configura logs detalhados
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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
            logger.info(f"Predição concluída: {prediction}")
        except Exception as e:
            logger.error(f"ERRO NA PREDIÇÃO: {str(e)}")
            raise
        
        # 3. Resposta
        response = {
            "statusCode": 200,
            "body": json.dumps({
                "prediction": int(prediction),
                "processing_time": str(datetime.now() - start_time)
            })
        }
        logger.info(f"Resposta gerada: {response}")
        return response
        
    except Exception as e:
        logger.error(f"ERRO NA LAMBDA: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }