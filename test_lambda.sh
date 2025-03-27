#!/bin/bash

# Configurações
MODEL_ID="2760004acc144318a4b77283a23bc827"
PORT=9000
CONTAINER_NAME="churn-prediction-test"

# 1. Verifica se o usuário tem permissão
if ! groups | grep -q '\bdocker\b'; then
    echo "ERRO: Seu usuário não está no grupo docker."
    echo "Execute estes comandos e depois reinicie seu terminal:"
    echo "  sudo usermod -aG docker $USER"
    echo "  newgrp docker"
    exit 1
fi

# 2. Constrói a imagem Docker
docker build -t churn-prediction-lambda --build-arg MODEL_ID=${MODEL_ID} . || {
    echo "ERRO ao construir a imagem Docker";
    exit 1;
}

# 3. Remove container existente (se houver)
docker rm -f ${CONTAINER_NAME} 2>/dev/null

# 4. Inicia o container
docker run -d -p ${PORT}:8080 --name ${CONTAINER_NAME} churn-prediction-lambda || {
    echo "ERRO ao iniciar o container";
    exit 1;
}

# 5. Aguarda o container estar pronto
echo "Aguardando o container iniciar..."
sleep 5

# 6. Testa a API com curl
echo "Testando a API..."
curl -XPOST "http://localhost:${PORT}/2015-03-31/functions/function/invocations" -d '{
    "tenure":24,
    "MonthlyCharges":65.75,
    "Partner":"Yes",
    "Dependents":"No",
    "InternetService":"Fiber optic",
    "OnlineSecurity":"No",
    "OnlineBackup":"Yes",
    "DeviceProtection":"No",
    "TechSupport":"No",
    "StreamingTV":"Yes",
    "StreamingMovies":"Yes",
    "Contract":"Two year",
    "PaymentMethod":"Credit card (automatic)",
    "QtdProducts":3
}'

# 7. Limpeza
echo "Parando o container..."
docker stop ${CONTAINER_NAME} 2>/dev/null
docker rm ${CONTAINER_NAME} 2>/dev/null